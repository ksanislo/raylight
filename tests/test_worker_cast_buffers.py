"""Workers release ComfyUI's weight-streaming state when they drop models.

ComfyUI resets it after every node in its own executor, which workers do not
run. Left alone, LARGEST_AIMDO_CASTED_WEIGHT holds the largest module ever
streamed, so a VAE dropped by the worker stays alive through that one module.
"""
import gc
import types
import weakref

import pytest
import torch

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

import comfy.model_management as mm
from raylight.distributed_worker import ray_worker


@pytest.fixture
def streamed_module(monkeypatch):
    module = torch.nn.Linear(4, 4)
    monkeypatch.setattr(mm, "LARGEST_AIMDO_CASTED_WEIGHT", (module, 1 << 30))
    monkeypatch.setattr(mm, "STREAM_AIMDO_CAST_BUFFERS", {"stream": object()})
    monkeypatch.setattr(mm, "current_loaded_models", [])
    return weakref.ref(module)


def _loaded(patcher_model):
    patcher = types.SimpleNamespace(model=patcher_model, is_dynamic=lambda: True)
    return types.SimpleNamespace(model=patcher)


def _forget(ref):
    gc.collect()
    return ref() is None


def test_a_dropped_module_is_not_kept_alive(streamed_module):
    ray_worker._reset_cast_buffers()

    assert _forget(streamed_module)
    assert mm.STREAM_AIMDO_CAST_BUFFERS == {}


def test_a_loaded_fsdp_model_does_not_stop_the_release(streamed_module, monkeypatch):
    # an FSDP patcher is dynamic but keeps no pin bookkeeping for the full reset to walk
    monkeypatch.setattr(mm, "current_loaded_models", [_loaded(torch.nn.Module())])

    def full_reset():
        raise AssertionError("the full reset reads dynamic_pins of every loaded dynamic model")

    monkeypatch.setattr(mm, "reset_cast_buffers", full_reset)

    ray_worker._reset_cast_buffers()

    assert _forget(streamed_module)
    assert mm.STREAM_AIMDO_CAST_BUFFERS == {}
    assert mm.LARGEST_AIMDO_CASTED_WEIGHT == (None, 0)
