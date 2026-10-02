import types

import pytest
import torch

import raylight.diffusion_models.minimax.fp16_support as fp16_support


@pytest.fixture
def comfyui(monkeypatch):
    """A MiniMax H3 config whose fp16 support each test sets, and a fresh lookup."""
    config = type("MiniMaxH3", (), {"supported_inference_dtypes": [torch.bfloat16, torch.float32]})
    monkeypatch.setattr(fp16_support, "_minimax_config", lambda: config)
    monkeypatch.setattr(fp16_support, "_HANDLED_BY_COMFYUI", None)
    monkeypatch.setenv("RAYLIGHT_FP32_RESIDUAL", "1")
    monkeypatch.setenv("RAYLIGHT_MLP_FP16", "1")
    return config


def test_comfyui_handles_fp16_when_it_offers_it_and_its_block_takes_the_branch_dtype(comfyui, monkeypatch):
    comfyui.supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    monkeypatch.setattr(fp16_support, "_block_takes_branch_dtype", lambda: True)

    assert fp16_support.comfyui_handles_fp16()
    assert not fp16_support.allow_fp16_inference({"is_xdit": True})


def test_offering_fp16_without_the_block_support_is_not_enough(comfyui, monkeypatch):
    comfyui.supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    monkeypatch.setattr(fp16_support, "_block_takes_branch_dtype", lambda: False)

    assert not fp16_support.comfyui_handles_fp16()


def test_raylight_adding_fp16_does_not_read_as_comfyui_support(comfyui, monkeypatch):
    monkeypatch.setattr(fp16_support, "_block_takes_branch_dtype", lambda: True)

    assert fp16_support.allow_fp16_inference({"is_xdit": True})
    assert torch.float16 in comfyui.supported_inference_dtypes
    assert not fp16_support.comfyui_handles_fp16()


def test_the_block_check_reads_the_installed_comfyui():
    # whichever ComfyUI this runs against, the lookup has to answer rather than raise
    assert fp16_support._block_takes_branch_dtype() in (True, False)


def _inject(native, monkeypatch):
    import sys

    import comfy.model_base
    from raylight.distributed_modules.usp import USPInjectRegistry

    # the real forwards build their sequence-parallel attention at import, which
    # needs process groups; only which of them the injector installs is checked here
    forwards = types.ModuleType("raylight.diffusion_models.minimax.xdit_context_parallel")
    for name in ("usp_attn_forward", "usp_block_forward", "usp_dit_forward", "usp_mlp_forward"):
        setattr(forwards, name, lambda self, *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, forwards.__name__, forwards)

    monkeypatch.setattr(fp16_support, "comfyui_handles_fp16", lambda: native)
    monkeypatch.setenv("RAYLIGHT_FP32_RESIDUAL", "1")
    monkeypatch.setenv("RAYLIGHT_MLP_FP16", "1")
    blocks = [types.SimpleNamespace(attn=types.SimpleNamespace(), mlp=types.SimpleNamespace(), forward="comfyui")
              for _ in range(2)]
    diffusion_model = types.SimpleNamespace(blocks=blocks, token_refiner=types.SimpleNamespace(blocks=[]))
    base_model = types.SimpleNamespace(diffusion_model=diffusion_model, extra_conds="comfyui",
                                       get_dtype_inference=lambda: torch.float16)
    patcher = types.SimpleNamespace(get_attachment=lambda key: {})
    USPInjectRegistry._REGISTRY[comfy.model_base.MiniMaxH3](patcher, base_model)
    return base_model


def test_native_fp16_keeps_comfyuis_block_mlp_and_conditioning(monkeypatch):
    base_model = _inject(True, monkeypatch)

    for block in base_model.diffusion_model.blocks:
        assert block.forward == "comfyui"
        assert not hasattr(block.mlp, "forward")
        assert hasattr(block.attn, "forward")
    assert base_model.extra_conds == "comfyui"


def test_without_native_fp16_raylight_installs_its_own(monkeypatch):
    base_model = _inject(False, monkeypatch)

    for block in base_model.diffusion_model.blocks:
        assert block.forward != "comfyui"
        assert hasattr(block.mlp, "forward")
    assert base_model.extra_conds != "comfyui"
