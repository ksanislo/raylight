import types

import pytest

import raylight.nodes as nodes


class _Actor:
    def __init__(self, calls, clear):
        def method(name, value=None):
            return types.SimpleNamespace(remote=lambda: (calls.append(name), value)[1])

        self.get_parallel_dict = method("get_parallel_dict", {"clear_vram_after_sampling": clear})
        self.free_cached_vae = method("free_cached_vae")
        self.clear_sampling_vram = method("clear_sampling_vram")


@pytest.fixture
def fake_ray(monkeypatch):
    monkeypatch.setattr(nodes, "ray", types.SimpleNamespace(get=lambda refs: refs))


@pytest.mark.parametrize("clear, expected", [
    (True, ["free_cached_vae"] * 2 + ["clear_sampling_vram"] * 2),
    (False, []),
])
def test_the_vae_is_released_after_a_decode_only_when_asked(fake_ray, clear, expected):
    calls = []
    actors = [_Actor(calls, clear) for _ in range(2)]

    nodes._free_ray_worker_vram_after_decode({"workers": actors})

    assert [c for c in calls if c != "get_parallel_dict"] == expected


def test_no_workers_is_a_no_op(fake_ray):
    nodes._free_ray_worker_vram_after_decode({"workers": []})
