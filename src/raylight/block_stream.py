"""Stream transformer block weights between host and device during the forward.

FSDP shards the model but keeps every shard resident, and the quantized shards are
built by hand after fully_shard() so neither CPUOffloadPolicy nor aimdo can evict
them. On a 16 GiB card that leaves ~58% of the card occupied by weights that are
each used once per step.

Each block's parameter storage is copied to host memory and released after the block
runs, and restored just before it runs again. The next block is restored ahead of
time on a side stream so the copy overlaps with compute.
"""

import os

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.utils import _alloc_storage, _free_storage


def enabled():
    return os.environ.get("RAYLIGHT_BLOCK_STREAM") == "1"


def _prefetch_depth():
    try:
        return max(1, int(os.environ.get("RAYLIGHT_BLOCK_STREAM_PREFETCH", "1")))
    except ValueError:
        return 1


def _storage_tensors(module):
    for m in module.modules():
        for p in m.parameters(recurse=False):
            data = p.data if isinstance(p.data, torch.Tensor) else None
            if data is None:
                continue
            inner = getattr(data, "_local_tensor", None) if isinstance(data, DTensor) else data
            if inner is None:
                continue
            qd = getattr(inner, "_qdata", None)
            real = qd if isinstance(qd, torch.Tensor) else inner
            if isinstance(real, torch.Tensor) and real.device.type == "cuda" and real.storage_offset() == 0:
                yield real


class BlockStreamer:
    def __init__(self, blocks, device):
        self.blocks = list(blocks)
        self.device = device
        self.host = {}          # block index -> [(tensor, host copy, size)]
        self.stream = torch.cuda.Stream(device=device)
        self.events = {}

    def offload(self, idx):
        if idx in self.host:
            return 0
        saved = []
        moved = 0
        for t in _storage_tensors(self.blocks[idx]):
            size = t.size()
            try:
                h = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
            except Exception:
                h = torch.empty(t.shape, dtype=t.dtype, device="cpu")
            h.copy_(t)
            try:
                _free_storage(t)
            except (RuntimeError, AssertionError):
                continue
            saved.append((t, h, size))
            moved += h.numel() * h.element_size()
        self.host[idx] = saved
        return moved

    def restore(self, idx, stream=None):
        saved = self.host.pop(idx, None)
        if not saved:
            return 0
        ctx = torch.cuda.stream(stream) if stream is not None else torch.cuda.stream(torch.cuda.current_stream(self.device))
        restored = 0
        with ctx:
            for t, h, size in saved:
                _alloc_storage(t, size)
                t.copy_(h, non_blocking=h.is_pinned())
                restored += h.numel() * h.element_size()
        if stream is not None:
            ev = torch.cuda.Event()
            ev.record(stream)
            self.events[idx] = ev
        return restored

    def wait(self, idx):
        ev = self.events.pop(idx, None)
        if ev is not None:
            torch.cuda.current_stream(self.device).wait_event(ev)

    def offload_all(self):
        return sum(self.offload(i) for i in range(len(self.blocks)))

    def install(self):
        n = len(self.blocks)
        depth = _prefetch_depth()

        def make_pre(i):
            def pre(_module, _args):
                self.wait(i)
                self.restore(i)
                for k in range(1, depth + 1):
                    nxt = i + k
                    if nxt < n and nxt in self.host:
                        self.stream.wait_stream(torch.cuda.current_stream(self.device))
                        self.restore(nxt, stream=self.stream)
            return pre

        def make_post(i):
            def post(_module, _args, _output):
                if i + depth < n or i == n - 1:
                    self.offload(i)
            return post

        for i, b in enumerate(self.blocks):
            b.register_forward_pre_hook(make_pre(i))
            b.register_forward_hook(make_post(i))
