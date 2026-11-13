import pytest
import torch

from raylight.comfy_dist.weight_adapter import lora as lora_module
from raylight.comfy_dist.weight_adapter.lora import (
    LoRAAdapter,
    release_bypass_operands,
    set_bypass_operand_placement,
)
from raylight.distributed_worker.raylightworker_config_schema import RayWorkerConfig


@pytest.fixture(autouse=True)
def _host_placement():
    set_bypass_operand_placement("host")
    yield
    set_bypass_operand_placement("host")


def _adapter(rank=4, in_dim=16, out_dim=24):
    torch.manual_seed(0)
    up, down = torch.randn(out_dim, rank), torch.randn(rank, in_dim)
    return LoRAAdapter(set(), (up, down, float(rank), None, None, None))


def _operands(adapter):
    return lora_module._bypass_operands(adapter, adapter.weights[0], adapter.weights[1],
                                        torch.device("cpu"), torch.float16)


def test_host_is_the_default_and_casts_every_forward():
    assert RayWorkerConfig().lora_bypass_operands == "host"
    adapter = _adapter()
    first, second = _operands(adapter), _operands(adapter)
    assert first[0] is not second[0]
    assert getattr(adapter, "_bypass_cache", None) is None


def test_resident_reuses_the_cast_until_released():
    set_bypass_operand_placement("resident")
    adapter = _adapter()
    first = _operands(adapter)
    assert _operands(adapter)[0] is first[0]
    assert first[0].dtype == torch.float16

    release_bypass_operands()
    assert adapter._bypass_cache is None
    assert _operands(adapter)[0] is not first[0]


def test_resident_casts_new_weights_again():
    set_bypass_operand_placement("resident")
    adapter = _adapter()
    first = _operands(adapter)
    up, down = adapter.weights[0].clone() * 2, adapter.weights[1]
    adapter.weights = (up, down) + tuple(adapter.weights[2:])
    second = _operands(adapter)
    assert second[0] is not first[0]
    assert torch.equal(second[0], up.half())


def test_changing_placement_drops_resident_copies():
    set_bypass_operand_placement("resident")
    adapter = _adapter()
    _operands(adapter)
    set_bypass_operand_placement("host")
    assert adapter._bypass_cache is None


def test_unknown_placement_is_rejected():
    with pytest.raises(ValueError):
        set_bypass_operand_placement("device")


def test_bypass_output_is_the_same_for_every_placement():
    x = torch.randn(5, 16)
    outputs = []
    for placement in ("host", "resident", "pinned"):
        set_bypass_operand_placement(placement)
        adapter = _adapter()
        if placement == "pinned":
            # page locking needs a CUDA driver; the CPU-only run keeps the host copy
            lora_module.logging.disable(lora_module.logging.WARNING)
        try:
            outputs.append(adapter.h(x, torch.zeros(5, 24)))
        finally:
            lora_module.logging.disable(lora_module.logging.NOTSET)
    assert torch.allclose(outputs[0], outputs[1]) and torch.allclose(outputs[0], outputs[2])
