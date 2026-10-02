import torch

from raylight.comfy_dist.sd import OffsetBypassAdapter
from raylight.comfy_dist.weight_adapter.lora import LoRAAdapter


def _adapter(entries=((4, 1.0), (8, 0.5), (2, 0.5)), in_dim=32, out_dim=48):
    torch.manual_seed(0)
    items = []
    for rank, strength in entries:
        up, down = torch.randn(out_dim, rank), torch.randn(rank, in_dim)
        lora = LoRAAdapter(set(), (up, down, float(rank), None, None, None))
        items.append({"adapter": lora, "offset": None, "strength": strength, "key": "k"})
    weight = torch.randn(out_dim, in_dim)
    return OffsetBypassAdapter(items), (lambda x: x @ weight.T)


def test_bypass_in_slices_matches_the_whole_output():
    adapter, forward = _adapter()
    x = torch.randn(1, 100, 32)
    expected = forward(x) + adapter.h(x, forward(x))

    adapter.BYPASS_CHUNK_ROWS = 7
    out = adapter.bypass_forward(forward, x)

    assert torch.allclose(out, expected, atol=1e-4)
    # the operands held for the slices are released afterwards
    assert all(getattr(item["adapter"], "_bypass_held", None) is None for item in adapter.entries)


def test_bypass_below_one_slice_takes_the_default_path():
    adapter, forward = _adapter()
    x = torch.randn(1, 100, 32)

    assert torch.equal(adapter.bypass_forward(forward, x), forward(x) + adapter.h(x, forward(x)))


def test_lora_sum_is_accumulated_in_place():
    adapter, forward = _adapter()
    x = torch.randn(100, 32)
    base = forward(x)
    expected = torch.zeros_like(base)
    for item in adapter.entries:
        up, down, alpha = item["adapter"].weights[:3]
        expected += (x @ down.T @ up.T) * (alpha / down.shape[0]) * item["strength"]

    assert torch.allclose(adapter.h(x, base), expected, atol=1e-4)
