import pathlib


def _lazy_host_vae_class():
    # nodes.py imports ComfyUI and Ray at module scope; the class itself needs
    # neither, so take its source alone
    source = (pathlib.Path(__file__).parent.parent / "src" / "raylight" / "nodes.py").read_text()
    start = source.index("class _LazyHostVAE:")
    end = source.index("class RayVAELoader:")
    namespace = {}
    exec(source[start:end], namespace)
    return namespace["_LazyHostVAE"]


class _FakeVAE:
    downscale_ratio = 8

    def encode(self, pixels):
        return ("encoded", pixels)


def test_nothing_is_loaded_until_the_vae_is_used():
    loads = []
    vae = _lazy_host_vae_class()("video.safetensors", loader=lambda name: loads.append(name) or _FakeVAE())

    assert loads == []
    assert vae.encode("frame") == ("encoded", "frame")
    assert loads == ["video.safetensors"]


def test_the_file_is_read_once_and_every_attribute_reaches_the_vae():
    loads = []
    vae = _lazy_host_vae_class()("video.safetensors", loader=lambda name: loads.append(name) or _FakeVAE())

    assert vae.downscale_ratio == 8
    vae.encode("a")
    vae.encode("b")
    assert loads == ["video.safetensors"]


def test_private_lookups_do_not_trigger_a_load():
    loads = []
    vae = _lazy_host_vae_class()("video.safetensors", loader=lambda name: loads.append(name) or _FakeVAE())

    assert not hasattr(vae, "_not_there")
    assert loads == []


def test_writes_land_on_the_loaded_vae():
    real = _FakeVAE()
    vae = _lazy_host_vae_class()("video.safetensors", loader=lambda name: real)

    vae.patcher = "replacement"
    assert real.patcher == "replacement"
    assert vae.patcher == "replacement"
