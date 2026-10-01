import logging
import torch

from .registry import register_inner_attention

try:
    import comfy_kitchen
except ImportError:
    comfy_kitchen = None


def int8_attention_is_available(device=None):
    if comfy_kitchen is None or not hasattr(comfy_kitchen, "int8_attention_is_available"):
        return False
    try:
        return bool(comfy_kitchen.int8_attention_is_available(device))
    except Exception:
        return False


def _values(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().reshape(-1).tolist()
    return list(value)


def _step_index(transformer_options):
    """Index of the current sampling step, or None when the schedule is not known."""
    schedule = _values(transformer_options.get("sample_sigmas"))[:-1]
    current = _values(transformer_options.get("sigmas"))
    if not schedule or not current:
        return None, len(schedule)
    try:
        index = min(range(len(schedule)), key=lambda i: abs(float(schedule[i]) - float(current[0])))
    except (TypeError, ValueError):
        return None, len(schedule)
    return index, len(schedule)


def _ring_size(group):
    if group is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group)


# Ring steps are merged by their log-sum-exp, which only Comfy Kitchen builds with
# int8_attention_with_lse return; without it the processor replaces the local
# attention of Ulysses alone.
@register_inner_attention("raylight:comfy_kitchen_int8")
class ComfyKitchenInt8Attention:
    supports_ring = comfy_kitchen is not None and hasattr(comfy_kitchen, "int8_attention_with_lse")

    def __init__(self, dense_first_steps=0):
        self.dense_first_steps = int(dense_first_steps)
        self._logged_step = None
        self._available = None
        self._logged_success = False
        self._logged_failure = False
        self._logged_skip = False

    def _is_available(self, device):
        if self._available is None:
            self._available = int8_attention_is_available(device)
        return self._available

    def __call__(self, dense_attention, q, k, v, *, transformer_options, **kwargs):
        if (kwargs.get("joint_tensor_key") is not None or kwargs.get("causal", False) or
                q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16) or
                q.ndim != 4 or not self._is_available(q.device)):
            if not self._logged_skip:
                logging.warning("[Raylight] Comfy Kitchen INT8 attention skipped; using dense attention "
                                "(device=%s dtype=%s ndim=%d available=%s)", q.device, q.dtype, q.ndim,
                                self._available)
                self._logged_skip = True
            return dense_attention(q, k, v, **kwargs)
        step, steps = _step_index(transformer_options)
        dense_step = self.dense_first_steps > 0 and (step is None or step < self.dense_first_steps)
        if step != self._logged_step:
            print("[Raylight] Comfy Kitchen INT8 step {}/{}: {}".format(
                "?" if step is None else step + 1, steps, "dense" if dense_step else "int8"), flush=True)
            self._logged_step = step
        if dense_step:
            return dense_attention(q, k, v, **kwargs)
        if _ring_size(kwargs.get("group")) > 1:
            return self._ring(dense_attention, q, k, v, kwargs)
        try:
            # xFuser hands [batch, seq, heads, dim]; the kernel takes [batch, heads, seq, dim].
            out = comfy_kitchen.int8_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=kwargs.get("softmax_scale", None),
            )
            out = out.transpose(1, 2).contiguous()
        except Exception as error:
            if not self._logged_failure:
                logging.warning("[Raylight] Comfy Kitchen INT8 attention failed; using dense attention (%s)", error)
                self._logged_failure = True
            return dense_attention(q, k, v, **kwargs)
        if not self._logged_success:
            print("[Raylight] Using custom attention: Comfy Kitchen INT8 q={} {}".format(
                tuple(q.shape), q.dtype), flush=True)
            self._logged_success = True
        return out

    def _ring(self, dense_attention, q, k, v, kwargs):
        # xFuser's ring loop runs one attention per key shard and merges the
        # results by log-sum-exp. An unrecognised attn_type makes it use
        # attn_processor for that per-shard attention.
        from yunchang.kernels import select_flash_attn_impl
        fallback = select_flash_attn_impl(kwargs.get("attn_type"), stage="fwd-only",
                                          attn_processor=kwargs.get("attn_processor"))

        def step(query, key, value, **step_kwargs):
            try:
                out, lse = comfy_kitchen.int8_attention_with_lse(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                    scale=step_kwargs.get("softmax_scale"),
                )
            except Exception as error:
                if not self._logged_failure:
                    logging.warning("[Raylight] Comfy Kitchen INT8 ring step failed; using dense attention (%s)",
                                    error)
                    self._logged_failure = True
                return fallback(query, key, value, **step_kwargs)
            if not self._logged_success:
                print("[Raylight] Using custom attention: Comfy Kitchen INT8 ring q={} {}".format(
                    tuple(query.shape), query.dtype), flush=True)
                self._logged_success = True
            return out.transpose(1, 2), lse

        options = dict(kwargs)
        options["attn_type"] = None
        options["attn_processor"] = step
        return dense_attention(q, k, v, **options)

