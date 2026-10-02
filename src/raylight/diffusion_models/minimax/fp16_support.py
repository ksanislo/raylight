"""fp16 inference for MiniMax H3 where ComfyUI does not offer it.

Volta and Turing have fp16 tensor cores but no bf16, and ComfyUI's MiniMax H3
config lists only bfloat16 and float32, so those cards run the model in fp32.
Raylight's USP forwards carry the residual in fp32 and rescale the two
projections that leave fp16's range, so under them fp16 is safe. This offers it
to the loader, and runs in fp32 the one piece of preprocessing that happens
outside those forwards.
"""
import os
import types

import torch


def _minimax_config():
    import comfy.supported_models

    return getattr(comfy.supported_models, "MiniMaxH3", None)


def fp16_forwards_active(parallel_dict):
    """Whether this worker installs the fp16-safe MiniMax H3 forwards.

    The block forward that keeps the residual in fp32 and the MLP forward that
    rescales fc2 are only installed under USP, and each behind its own switch.
    Without both, fc2's output (~3.7e6) and the residual (~1e7) overflow fp16.
    """
    return (bool(parallel_dict.get("is_xdit"))
            and os.environ.get("RAYLIGHT_FP32_RESIDUAL") == "1"
            and os.environ.get("RAYLIGHT_MLP_FP16") == "1")


def allow_fp16_inference(parallel_dict):
    """Let the loader choose fp16 for MiniMax H3 when this worker can run it safely.

    The compute dtype comes from the config's supported_inference_dtypes, so fp16
    is added there rather than forced as the weight dtype, which would leave the
    model casting up to fp32. bfloat16 stays first, so a card that has it is
    unaffected. Only this worker process sees the change.
    """
    config = _minimax_config()
    if config is None or not fp16_forwards_active(parallel_dict):
        return False
    dtypes = list(config.supported_inference_dtypes)
    if torch.float16 in dtypes:
        return False
    position = dtypes.index(torch.float32) if torch.float32 in dtypes else len(dtypes)
    dtypes.insert(position, torch.float16)
    config.supported_inference_dtypes = dtypes
    return True


def preprocess_text_in_fp32(base_model):
    """Run MiniMax H3's text preprocessing in fp32 when the model infers in fp16.

    extra_conds casts the Qwen3-VL states to the inference dtype before
    condition_proj, and they reach ~9.6e4, past fp16's 65504. They are refined here
    in fp32 instead and only the result is cast down; preprocess_text_embeds then
    passes the already-refined states through unchanged.
    """
    if getattr(base_model, "_raylight_fp32_text", False):
        return
    original = base_model.extra_conds

    def extra_conds(self, **kwargs):
        cross_attn = kwargs.get("cross_attn")
        dtype = self.get_dtype_inference()
        if cross_attn is not None and dtype == torch.float16:
            text = cross_attn.to(device=kwargs["device"], dtype=torch.float32)
            kwargs = dict(kwargs, cross_attn=self.diffusion_model.preprocess_text_embeds(text).to(dtype))
        return original(**kwargs)

    base_model.extra_conds = types.MethodType(extra_conds, base_model)
    base_model._raylight_fp32_text = True
