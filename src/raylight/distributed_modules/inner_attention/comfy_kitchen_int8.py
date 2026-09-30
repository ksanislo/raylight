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


# Comfy Kitchen's INT8 attention returns no log-sum-exp, so the partial results of
# a ring step cannot be merged; it only replaces the local attention of Ulysses.
@register_inner_attention("raylight:comfy_kitchen_int8")
class ComfyKitchenInt8Attention:
    supports_ring = False

    def __init__(self):
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
