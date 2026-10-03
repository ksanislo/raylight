import types

import pytest

import raylight.nodes as nodes


class _Actor:
    def __init__(self, calls, clear):
        def method(name, value=None):
            return types.SimpleNamespace(remote=lambda: (calls.append(name), value)[1])

        self.get_parallel_dict = method("get_parallel_dict", {"clear_vram_after_sampling": clear})
        self.free_cached_vae = method("free_cached_vae")
        self.free_cached_audio_vae = method("free_cached_audio_vae")
        self.clear_sampling_vram = method("clear_sampling_vram")


@pytest.fixture
def fake_ray(monkeypatch):
    monkeypatch.setattr(nodes, "ray", types.SimpleNamespace(get=lambda refs: refs,
                                                            wait=lambda refs, timeout=None: (refs, [])))


@pytest.mark.parametrize("clear, expected", [
    (True, ["free_cached_vae"] * 2 + ["free_cached_audio_vae"] * 2 + ["clear_sampling_vram"] * 2),
    (False, []),
])
def test_the_vae_is_released_after_a_decode_only_when_asked(fake_ray, clear, expected):
    calls = []
    actors = [_Actor(calls, clear) for _ in range(2)]

    nodes._free_ray_worker_vram_after_decode({"workers": actors})

    assert [c for c in calls if c != "get_parallel_dict"] == expected


def test_no_workers_is_a_no_op(fake_ray):
    nodes._free_ray_worker_vram_after_decode({"workers": []})


def test_audio_decodes_only_the_audio_stream_on_the_last_worker(fake_ray, monkeypatch):
    import torch
    import comfy.nested_tensor

    monkeypatch.setattr(nodes.folder_paths, "get_full_path_or_raise", lambda kind, name: f"/vae/{name}")
    calls, jobs = [], []
    actors = [_Actor(calls, clear=True) for _ in range(3)]
    actors[-1].ray_audio_vae_decode = types.SimpleNamespace(
        remote=lambda path, job: (jobs.append((path, job)), {"waveform": "audio", "sample_rate": 48000})[1])
    video, audio = torch.zeros(1, 4, 2, 2), torch.ones(1, 8, 16)
    samples = {"samples": comfy.nested_tensor.NestedTensor([video, audio])}

    out = nodes.RayVAEDecodeAudio().ray_decode_audio({"workers": actors}, samples, "audio.safetensors")

    assert out == ({"waveform": "audio", "sample_rate": 48000},)
    path, job = jobs[0]
    assert path == "/vae/audio.safetensors"
    assert torch.equal(job["samples"], audio)
    assert calls.count("free_cached_audio_vae") == 3
