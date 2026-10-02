import types

import pytest
import torch

import raylight.diffusion_models.minimax.fp16_support as fp16_support


SAFE = {"RAYLIGHT_FP32_RESIDUAL": "1", "RAYLIGHT_MLP_FP16": "1"}


@pytest.fixture
def config(monkeypatch):
    stock = type("MiniMaxH3", (), {"supported_inference_dtypes": [torch.bfloat16, torch.float32]})
    monkeypatch.setattr(fp16_support, "_minimax_config", lambda: stock)
    return stock


def _env(monkeypatch, values):
    for name in ("RAYLIGHT_FP32_RESIDUAL", "RAYLIGHT_MLP_FP16"):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_fp16_is_offered_after_bf16_when_the_forwards_are_safe(config, monkeypatch):
    _env(monkeypatch, SAFE)

    assert fp16_support.allow_fp16_inference({"is_xdit": True})
    assert config.supported_inference_dtypes == [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize("parallel_dict, env", [
    ({"is_xdit": False}, SAFE),
    ({"is_xdit": True}, {"RAYLIGHT_FP32_RESIDUAL": "1"}),
    ({"is_xdit": True}, {"RAYLIGHT_MLP_FP16": "1"}),
])
def test_fp16_is_not_offered_without_the_safe_forwards(config, monkeypatch, parallel_dict, env):
    _env(monkeypatch, env)

    assert not fp16_support.allow_fp16_inference(parallel_dict)
    assert config.supported_inference_dtypes == [torch.bfloat16, torch.float32]


def test_a_config_that_already_offers_fp16_is_left_alone(config, monkeypatch):
    _env(monkeypatch, SAFE)
    config.supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    assert not fp16_support.allow_fp16_inference({"is_xdit": True})


def _model(native):
    """A MiniMax H3 base model whose text preprocessing overflows fp16 on the way."""

    def preprocess_text_embeds(text):
        # like the real one: text_dim 8 -> hidden 4, and hidden-size states pass through
        if text.shape[-1] == 4:
            return text
        return text[..., :4] / 4.0

    def extra_conds(self, **kwargs):
        cross_attn = kwargs["cross_attn"]
        dtype = self.get_dtype_inference()
        if native:
            cross_attn = preprocess_text_embeds(cross_attn.to(torch.float32)).to(dtype)
        else:
            cross_attn = preprocess_text_embeds(cross_attn.to(dtype))
        return {"c_crossattn": cross_attn}

    model = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(preprocess_text_embeds=preprocess_text_embeds))
    model.get_dtype_inference = lambda: torch.float16
    model.extra_conds = types.MethodType(extra_conds, model)
    return model


def test_stock_preprocessing_overflows_and_the_fp32_wrap_does_not():
    states = torch.full((1, 4, 8), 9.6e4)
    stock = _model(native=False)
    assert not torch.isfinite(stock.extra_conds(cross_attn=states, device="cpu")["c_crossattn"]).all()

    fp16_support.preprocess_text_in_fp32(stock)
    out = stock.extra_conds(cross_attn=states, device="cpu")["c_crossattn"]

    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_the_wrap_is_harmless_where_comfyui_already_preprocesses_in_fp32():
    states = torch.full((1, 4, 8), 9.6e4)
    native = _model(native=True)
    expected = native.extra_conds(cross_attn=states, device="cpu")["c_crossattn"]

    fp16_support.preprocess_text_in_fp32(native)

    assert torch.equal(native.extra_conds(cross_attn=states, device="cpu")["c_crossattn"], expected)


def test_the_wrap_is_applied_once():
    model = _model(native=False)
    fp16_support.preprocess_text_in_fp32(model)
    wrapped = model.extra_conds
    fp16_support.preprocess_text_in_fp32(model)

    assert model.extra_conds is wrapped
