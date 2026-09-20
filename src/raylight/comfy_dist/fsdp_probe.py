"""Measurements for the FSDP all-gather path.

Nothing here is imported unless RAYLIGHT_FSDP_PROBE is set, and the probes cost
a Python call plus a handful of integer adds per collective. They are still
measurement scaffolding: never quote a timing from a build that has them on.

Probe A  counts the bytes each all-gather actually moves, and in what dtype.
Probe B  a one-off census of how the model was sharded.
"""

import collections
import logging
import os
import time

import torch

_orig_foreach_all_gather = None
_orig_prefetch_unshard = None
_orig_unshard = None
_stats = None
_profiler = None
_sample_started = None


def enabled():
    return os.environ.get("RAYLIGHT_FSDP_PROBE") == "1"


def _new_stats():
    return {
        "collectives": 0,
        "bytes_out": 0,
        "bytes_by_dtype": collections.Counter(),
        "out_dtype": collections.Counter(),
        "params": 0,
        "prefetches": 0,
        "unshards": 0,
    }


def install():
    """Wrap foreach_all_gather to record volume and dtype per collective."""
    global _orig_foreach_all_gather, _stats
    if _orig_foreach_all_gather is not None:
        return
    # The name is bound into _fsdp_param_group at import, so patching
    # _fsdp_collectives alone would miss the call site that matters.
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

    _stats = _new_stats()
    global _profiler
    if profiling():
        _profiler = _Profiler()
    h2d = int(os.environ.get("RAYLIGHT_FSDP_H2D_TRACE", "0"))
    if h2d > 0:
        _record_h2d(h2d)
    _time_forwards()
    _orig_foreach_all_gather = pg.foreach_all_gather

    def counting_foreach_all_gather(*args, **kwargs):
        result = _orig_foreach_all_gather(*args, **kwargs)
        try:
            out = result.all_gather_output
            _stats["collectives"] += 1
            if _profiler is not None:
                _profiler.tick(_stats["collectives"])
            _stats["bytes_out"] += out.numel() * out.element_size()
            _stats["out_dtype"][str(out.dtype).replace("torch.", "")] += 1
            # Both are list[list[...]], one inner list per parameter.
            for numels, dtypes in zip(
                result.param_all_gather_input_numels,
                result.param_all_gather_input_dtypes,
            ):
                _stats["params"] += 1
                for numel, dtype in zip(numels, dtypes):
                    _stats["bytes_by_dtype"][str(dtype).replace("torch.", "")] += (
                        numel * dtype.itemsize
                    )
        except Exception:
            pass
        return result

    pg.foreach_all_gather = counting_foreach_all_gather

    # Did the forward-prefetch path actually fire, and how often relative to
    # the plain unshards? Distinguishes "configured but never called" from
    # "called and made no difference".
    global _orig_prefetch_unshard
    _orig_prefetch_unshard = pg.FSDPParamGroup._prefetch_unshard

    @staticmethod
    def counting_prefetch_unshard(target, pass_type):
        if _stats is not None:
            _stats["prefetches"] += 1
        return _orig_prefetch_unshard(target, pass_type)

    pg.FSDPParamGroup._prefetch_unshard = counting_prefetch_unshard

    global _orig_unshard
    _orig_unshard = pg.FSDPParamGroup.unshard

    def counting_unshard(self, *a, **kw):
        if _stats is not None:
            _stats["unshards"] += 1
        return _orig_unshard(self, *a, **kw)

    pg.FSDPParamGroup.unshard = counting_unshard


def uninstall():
    global _orig_foreach_all_gather
    if _orig_foreach_all_gather is None:
        return
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group as pg

    pg.foreach_all_gather = _orig_foreach_all_gather
    _orig_foreach_all_gather = None


def reset():
    global _stats, _sample_started
    _stats = _new_stats()
    _sample_started = time.perf_counter()
    if _h2d_sites is not None:
        _h2d_sites.clear()
        _h2d_calls.clear()
    if _profiler is not None and _profiler.whole_sample:
        _profiler.begin()


def report(rank, tag):
    if _stats is None:
        return
    if _sample_started is not None:
        print(
            f"[Raylight][SAMPLE rank{rank}] wall={time.perf_counter() - _sample_started:.1f}s",
            flush=True,
        )
    mib = _stats["bytes_out"] / 2**20
    by_dtype = " ".join(
        f"{d}={b / 2**20:.0f}MiB" for d, b in _stats["bytes_by_dtype"].most_common()
    )
    outs = " ".join(f"{d}x{n}" for d, n in _stats["out_dtype"].most_common())
    print(
        f"[Raylight][FSDP-PROBE rank{rank} {tag}] collectives={_stats['collectives']} "
        f"gathered={mib:.0f} MiB params={_stats['params']} "
        f"unshards={_stats['unshards']} prefetches={_stats['prefetches']} "
        f"out_dtype=[{outs}] "
        f"| inputs: {by_dtype}",
        flush=True,
    )
    if _profiler is not None and _profiler.whole_sample:
        _profiler.finish()
    report_forwards(rank)
    report_h2d(rank)


def census(diffusion_model, rank):
    """One-off: how the model was actually sharded."""
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    groups = 0
    params = 0
    sharded_bytes = 0
    mismatched = 0
    no_extension = 0
    kinds = collections.Counter()

    for module in diffusion_model.modules():
        if not isinstance(module, FSDPModule):
            continue
        state = _get_module_fsdp_state(module)
        group = getattr(state, "_fsdp_param_group", None) if state is not None else None
        if group is None:
            continue
        groups += 1
        for p in group.fsdp_params:
            params += 1
            local = p._sharded_local_tensor
            kinds[type(local).__name__ + ":" + str(getattr(local, "_layout_cls", "-"))] += 1
            try:
                # A QuantizedTensor reports orig_dtype (bf16), not the int8 it
                # actually stores, so measure the real buffer.
                inner = getattr(local, "_qdata", None)
                real = inner if isinstance(inner, torch.Tensor) else local
                sharded_bytes += real.numel() * real.element_size()
            except Exception:
                pass
            if tuple(local.size()) != tuple(p.padded_sharded_param_size):
                mismatched += 1
            if not hasattr(local, "fsdp_pre_all_gather"):
                no_extension += 1

    print(
        f"[Raylight][FSDP-CENSUS rank{rank}] groups={groups} params={params} "
        f"sharded={sharded_bytes / 2**20:.0f} MiB "
        f"padded_mismatch={mismatched} without_pre_all_gather={no_extension}",
        flush=True,
    )
    for kind, n in kinds.most_common(6):
        print(f"[Raylight][FSDP-CENSUS rank{rank}]   {n:4d} x {kind}", flush=True)


def profiling():
    return os.environ.get("RAYLIGHT_FSDP_PROFILE") == "1"


class _Profiler:
    """Probe C: where the time in one warm forward actually goes.

    Driven off the all-gather counter rather than wrapping the whole sample, so
    it covers a single forward well after warm-up: cold-start costs land in the
    first forward and would otherwise be averaged into the table.

    FSDP emits its own record_function markers (FSDP::pre_forward,
    FSDP::all_gather, ...), so cost is attributed to the phase that caused it.
    Device time in nccl kernels against wall time says whether the collectives
    overlap compute or sit exposed.
    """

    def __init__(self):
        spec = os.environ.get("RAYLIGHT_FSDP_PROFILE_RANGE", "160:320")
        # Without sharding there are no all-gathers to count, so the window has
        # to be the whole sample.
        self.whole_sample = spec == "all"
        lo, _, hi = spec.partition(":")
        self.start_at = 0 if self.whole_sample else int(lo)
        self.stop_at = 0 if self.whole_sample else int(hi)
        self.rank = int(os.environ.get("RANK", "0"))
        self.profiler = None
        self.started = None
        self.done = False

    def begin(self):
        self.tick(self.start_at)

    def finish(self):
        if self.profiler is not None and self.started is not None:
            wall = time.perf_counter() - self.started
            self.profiler.stop()
            self.done = True
            self._report(wall, -1)

    def tick(self, count):
        if self.done:
            return
        if self.profiler is None and count >= self.start_at:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profiler.start()
            self.started = time.perf_counter()
        elif self.profiler is not None and not self.whole_sample and count >= self.stop_at:
            wall = time.perf_counter() - self.started
            self.profiler.stop()
            self.done = True
            self._report(wall, count)

    def _emit(self, text):
        print(text, flush=True)
        path = os.environ.get("RAYLIGHT_FSDP_PROF_FILE")
        if path:
            with open(path.replace("%r", str(self.rank)), "a") as fh:
                fh.write(text + "\n")

    def _report(self, wall, count):
        events = self.profiler.key_averages()
        nccl_us = sum(e.self_device_time_total for e in events if "nccl" in e.key.lower())
        device_us = sum(e.self_device_time_total for e in events)
        cpu_us = sum(e.self_cpu_time_total for e in events)
        self._emit(
            f"[Raylight][FSDP-PROF rank{self.rank}] collectives {self.start_at}-{count} "
            f"wall={wall:.1f}s device={device_us / 1e6:.1f}s cpu={cpu_us / 1e6:.1f}s "
            f"nccl={nccl_us / 1e6:.1f}s "
            f"({100 * nccl_us / max(device_us, 1):.1f}% of device, "
            f"{100 * nccl_us / max(wall * 1e6, 1):.1f}% of wall)"
        )
        self._emit(events.table(sort_by="self_device_time_total", row_limit=30))
        self._emit(events.table(sort_by="self_cpu_time_total", row_limit=25))
        path = os.environ.get("RAYLIGHT_FSDP_TRACE")
        if path:
            out = path.replace("%r", str(self.rank))
            self.profiler.export_chrome_trace(out)
            print(f"[Raylight][FSDP-PROF rank{self.rank}] trace -> {out}", flush=True)
        self.profiler = None


_h2d_sites = None
_h2d_calls = None
_orig_to = None
_orig_copy_ = None


def _record_h2d(limit):
    """Attribute host->device copies to the python frame that asked for them.

    A pageable H2D copy blocks the calling thread, which costs far more than the
    transfer itself and stops the host running ahead to enqueue later work.
    Counting them by call site says which code to fix.
    """
    import traceback

    global _h2d_sites, _h2d_calls, _orig_to
    if _orig_to is not None:
        return
    _h2d_sites = collections.Counter()
    _h2d_calls = collections.Counter()
    _orig_to = torch.Tensor.to

    def traced_to(self, *args, **kwargs):
        result = _orig_to(self, *args, **kwargs)
        if (
            self.device.type == "cpu"
            and isinstance(result, torch.Tensor)
            and result.device.type == "cuda"
            and sum(_h2d_calls.values()) < limit
        ):
            frames = [
                f"{os.path.basename(f.filename)}:{f.lineno} {f.name}"
                for f in traceback.extract_stack()[:-1]
                if "/torch/" not in f.filename
            ]
            site = " <- ".join(reversed(frames[-4:]))
            _h2d_sites[site] += self.numel() * self.element_size()
            _h2d_calls[site] += 1
        return result

    torch.Tensor.to = traced_to

    # ComfyUI's weight cast path copies into a preallocated destination rather
    # than calling .to(), so copy_ has to be counted separately or the bulk of
    # the streaming traffic is invisible.
    global _orig_copy_
    _orig_copy_ = torch.Tensor.copy_

    def traced_copy_(self, src, *args, **kwargs):
        if (
            isinstance(src, torch.Tensor)
            and src.device.type == "cpu"
            and self.device.type == "cuda"
            and sum(_h2d_calls.values()) < limit
        ):
            frames = [
                f"{os.path.basename(f.filename)}:{f.lineno} {f.name}"
                for f in traceback.extract_stack()[:-1]
                if "/torch/" not in f.filename
            ]
            site = "copy_ " + " <- ".join(reversed(frames[-4:]))
            _h2d_sites[site] += src.numel() * src.element_size()
            _h2d_calls[site] += 1
        return _orig_copy_(self, src, *args, **kwargs)

    torch.Tensor.copy_ = traced_copy_


def report_h2d(rank):
    if not _h2d_sites:
        return
    print(f"[Raylight][H2D rank{rank}] python-level host->device copies by site:", flush=True)
    print(
        f"[Raylight][H2D rank{rank}] total {sum(_h2d_calls.values())} calls "
        f"{sum(_h2d_sites.values()) / 2**20:.1f} MiB",
        flush=True,
    )
    for site, nbytes in _h2d_sites.most_common(12):
        print(
            f"[Raylight][H2D rank{rank}]   {nbytes / 2**20:8.2f} MiB "
            f"{_h2d_calls[site]:6d} calls  {site}",
            flush=True,
        )


_forward_times = None
_orig_apply_model = None


def _time_forwards():
    """Wall time of each denoise forward.

    The sampler bracket also covers loading and conditioning, so it cannot be
    divided by the step count to get a per step figure. This hook sits on the
    model call itself and is present whether or not sharding is on.
    """
    global _forward_times, _orig_apply_model
    if _orig_apply_model is not None:
        return
    import comfy.model_base

    _forward_times = []
    _orig_apply_model = comfy.model_base.BaseModel._apply_model

    def timed_apply_model(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return _orig_apply_model(self, *args, **kwargs)
        finally:
            _forward_times.append(time.perf_counter() - start)

    comfy.model_base.BaseModel._apply_model = timed_apply_model


def report_forwards(rank):
    if not _forward_times:
        return
    times = list(_forward_times)
    _forward_times.clear()
    joined = " ".join(f"{t:.1f}" for t in times)
    print(
        f"[Raylight][FORWARD rank{rank}] n={len(times)} "
        f"mean={sum(times) / len(times):.2f}s  [{joined}]",
        flush=True,
    )
