"""A sharded VAE decoder decodes exactly what an unsharded one does.

Runs CPU ranks over gloo with a small MiniMax H3 VAE: chunk counts that do not
divide by the rank count, so a rank makes a spare decoder call to stay in step;
tiles small enough that the tile batching is consulted; three ranks, so the
shards are padded; and the shards parked on the host between two decodes.
"""
import os
import socket
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
pytest.importorskip("comfy.ldm.minimax.vae")

CASES = [(2, 17), (3, 22)]  # (ranks, latent frames)


def _latent(frames):
    torch.manual_seed(1)
    return torch.randn(1, 24, frames, 6, 6)


def _small_vae():
    import comfy.ldm.minimax.vae as minimax

    torch.manual_seed(0)
    model = minimax.MiniMaxH3VideoVAE(ch=32, ch_mult=(1, 1, 1, 1, 1, 1))
    model.decoder = minimax.ViT3DDecoder(patch_size=16, patch_size_t=4, in_channels=24, num_layers=3, heads=2, dim_head=16)
    model.tile_size = 64
    model.tile_overlap_min = 16
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    model.eval()
    vae = types.SimpleNamespace(
        first_stage_model=model,
        patcher=types.SimpleNamespace(size=0, model_size=lambda: 0),
        size=None,
        vae_dtype=torch.float32,
        device=torch.device("cpu"),
        disable_offload=True,
        memory_used_decode=lambda shape, dtype: 0,
    )
    return vae


def _decode(vae, latent, rank, world):
    from raylight.distributed_worker import ray_worker_vae

    worker = types.SimpleNamespace(vae_model=vae)
    return ray_worker_vae.ray_vae_decode_temporal_partial_impl(
        worker, {"samples": latent}, job_rank=rank, job_world_size=world)


def _patch_comfy():
    import comfy.model_management as mm

    mm.load_models_gpu = lambda *a, **k: None


def _rank(rank, world, frames, port, tmpdir, queue):
    os.environ["RAYLIGHT_RAY_TMPDIR"] = tmpdir
    # the comparison is bit for bit, so the ranks reduce in the same order as the
    # single threaded reference
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}")
    try:
        _patch_comfy()
        from torch.distributed.tensor import DTensor
        from raylight.distributed_worker import ray_worker_vae

        vae = _small_vae()
        ray_worker_vae.shard_vae_decoder(vae, dist.device_mesh.init_device_mesh("cpu", (world,)))
        ray_worker_vae.offload_sharded_decoder(vae)
        # ComfyUI's patcher walks the module tree and must never meet a shard
        assert not any(isinstance(p, DTensor) for p in vae.first_stage_model.parameters())
        weight = vae.first_stage_model.decoder.transformer_blocks[0].attn.to_qkv.weight
        assert isinstance(weight, DTensor) and weight.to_local().shape[0] < weight.shape[0]
        assert weight.to_local().untyped_storage().nbytes() == 0

        latent = _latent(frames)
        first = _decode(vae, latent, rank, world)
        assert weight.to_local().untyped_storage().nbytes() == 0
        second = _decode(vae, latent, rank, world)
        for (i, a), (j, b) in zip(first["chunks"], second["chunks"]):
            assert i == j and torch.equal(a, b)
        queue.put((rank, first["num_chunks"], [(i, c.numpy()) for i, c in first["chunks"]]))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world, frames", CASES)
def test_sharded_decode_matches_unsharded(tmp_path, monkeypatch, world, frames):
    monkeypatch.setenv("RAYLIGHT_RAY_TMPDIR", str(tmp_path))
    torch.set_num_threads(1)
    _patch_comfy()

    expected = _decode(_small_vae(), _latent(frames), 0, 1)
    assert expected["num_chunks"] % world != 0, "needs a spare call to be exercised"

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_rank, args=(r, world, frames, port, str(tmp_path), queue)) for r in range(world)]
    for p in procs:
        p.start()
    try:
        # a rank out of step hangs in a collective rather than failing
        results = [queue.get(timeout=300) for _ in procs]
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()

    got = {}
    for rank, num_chunks, chunks in results:
        assert num_chunks == expected["num_chunks"]
        assert [i for i, _ in chunks] == list(range(rank, num_chunks, world))
        got.update({i: torch.from_numpy(c) for i, c in chunks})
    assert sorted(got) == list(range(expected["num_chunks"]))
    for i, chunk in expected["chunks"]:
        assert torch.equal(got[i], chunk), f"chunk {i} differs: max {(got[i] - chunk).abs().max()}"


def test_the_loader_hands_the_shard_setting_to_every_worker(monkeypatch):
    import raylight.nodes as nodes

    calls = []

    class _Actor:
        ray_vae_loader = types.SimpleNamespace(remote=lambda path, shard: calls.append((path, shard)))

    monkeypatch.setattr(nodes, "ray", types.SimpleNamespace(get=lambda refs: refs))
    monkeypatch.setattr(nodes.folder_paths, "get_full_path_or_raise", lambda kind, name: f"/vae/{name}")

    ray_vae, _ = nodes.RayVAELoader().load_vae({"workers": [_Actor(), _Actor()]}, "video.safetensors", True)

    assert ray_vae == {"vae_path": "/vae/video.safetensors", "shard_weights": True}
    assert calls == [("/vae/video.safetensors", True)] * 2
