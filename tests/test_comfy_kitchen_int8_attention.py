import sys
import types

import pytest
import torch

import raylight.distributed_modules.inner_attention.comfy_kitchen_int8 as ck_int8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel path only runs on CUDA")


def _qkv():
    return [torch.zeros(1, 8, 2, 16, dtype=torch.float16, device="cuda") for _ in range(3)]


def _processor(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(ck_int8, "comfy_kitchen", types.SimpleNamespace(
        int8_attention=fail, int8_attention_with_lse=fail))
    processor = ck_int8.ComfyKitchenInt8Attention()
    processor._available = True
    return processor


def _dense(*args, **kwargs):
    return "dense"


def test_an_unsupported_input_falls_back_to_dense(monkeypatch):
    processor = _processor(monkeypatch, RuntimeError("unsupported head dim"))

    assert processor(_dense, *_qkv(), transformer_options={}) == "dense"


def test_out_of_memory_is_raised_rather_than_retried_dense(monkeypatch):
    processor = _processor(monkeypatch, torch.cuda.OutOfMemoryError("out of memory"))

    with pytest.raises(torch.cuda.OutOfMemoryError):
        processor(_dense, *_qkv(), transformer_options={})


@pytest.mark.parametrize("error, expected", [
    (RuntimeError("unsupported head dim"), "fallback"),
    (torch.cuda.OutOfMemoryError("out of memory"), torch.cuda.OutOfMemoryError),
])
def test_ring_step_falls_back_except_on_out_of_memory(monkeypatch, error, expected):
    processor = _processor(monkeypatch, error)
    kernels = types.ModuleType("yunchang.kernels")
    kernels.select_flash_attn_impl = lambda *args, **kwargs: (lambda *a, **k: "fallback")
    monkeypatch.setitem(sys.modules, "yunchang.kernels", kernels)

    def ring(q, k, v, attn_processor=None, **kwargs):
        # what xFuser's ring loop does with an unrecognised attn_type: one step per key shard
        return attn_processor(q, k, v)

    if expected == "fallback":
        assert processor._ring(ring, *_qkv(), {}) == "fallback"
    else:
        with pytest.raises(expected):
            processor._ring(ring, *_qkv(), {})
