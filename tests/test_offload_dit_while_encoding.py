"""The diffusion model steps off the card while the text encoder loads and runs.

With the diffusion model resident the two compete for the card: the encoder's
load decides from free memory whether its shards can live in vram, and on a
small card the answer is no while the diffusion model sits there.
"""
import types

import pytest
import torch

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

import comfy.sd
import raylight.comfy_dist.sd as dist_sd
from raylight.distributed_worker.ray_worker import RayWorker


class _Model:
    def __init__(self, log):
        self.log = log

    def offload_fsdp_vram(self):
        self.log.append("dit off")
        return 1


class _Clip:
    def __init__(self, log):
        self.log = log

    def load_model(self, tokens):
        self.log.append("encoder on")


def _worker(log, model="fsdp"):
    w = RayWorker.__new__(RayWorker)
    w.local_rank = 0
    w.model = _Model(log) if model == "fsdp" else model
    w.clip = None
    w.clip_key = None
    w.clip_patcher = None
    w.clip_state_dict = None
    w.device_mesh = None
    w.is_cpu_offload = False
    w.parallel_dict = {}
    w.restore_clip_vram = lambda: log.append("encoder restored")
    w._free_current_clip = lambda: None
    return w


@pytest.fixture
def loader(monkeypatch):
    def install(log):
        def load(*args, **kwargs):
            log.append("encoder fit check")
            return _Clip(log), object(), None

        monkeypatch.setattr(dist_sd, "fsdp_load_text_encoder", load)

    return install


def _clip_type():
    return comfy.sd.CLIPType.MINIMAX.value


def test_the_load_measures_with_the_diffusion_model_off_the_card(loader):
    log = []
    loader(log)
    w = _worker(log)

    w.load_clip("/te.safetensors", _clip_type())

    assert log == ["dit off", "encoder fit check"]


def test_a_cached_encoder_moves_nothing(loader):
    log = []
    loader(log)
    w = _worker(log)
    w.load_clip("/te.safetensors", _clip_type())
    log.clear()

    w.load_clip("/te.safetensors", _clip_type())

    assert log == []


def test_the_encode_runs_with_the_diffusion_model_off_the_card(loader):
    log = []
    loader(log)
    w = _worker(log)
    w.load_clip("/te.safetensors", _clip_type())
    log.clear()

    w.prepare_clip({"tokens": []})

    assert log.index("dit off") < log.index("encoder restored") < log.index("encoder on")


@pytest.mark.parametrize("model", [None, types.SimpleNamespace()])
def test_a_model_without_fsdp_shards_is_left_alone(loader, model):
    log = []
    loader(log)
    w = _worker(log, model=model)

    w.load_clip("/te.safetensors", _clip_type())
    w.prepare_clip({"tokens": []})

    assert "dit off" not in log
