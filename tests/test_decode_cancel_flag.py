"""A cancelled distributed decode does not leave its cancel request behind.

The request is a flag on the progress board that the workers poll. A sampling
worker clears it as it stops, but a decode's workers only read it, so the host
has to take it back once they have answered - or every later decode stops at its
first chunk.
"""
import types

import pytest
import torch

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

import comfy.model_management
import raylight.nodes as nodes
from raylight import progress
import raylight.comfy_extra_dist.nodes_custom_sampler as custom_sampler


@pytest.fixture
def board():
    board = progress.Board()
    progress.use_local_board(board)
    progress.clear()
    yield board
    progress.use_local_board(None)


def _interrupted(monkeypatch, seen):
    monkeypatch.setattr(nodes, "ray", types.SimpleNamespace(
        wait=lambda pending, num_returns, timeout: ([], pending),
        get=lambda refs: refs))

    def interrupt():
        raise comfy.model_management.InterruptProcessingException()

    monkeypatch.setattr(comfy.model_management, "throw_exception_if_processing_interrupted", interrupt)

    def drain(futures, _checkpoint):
        # the workers poll the request while they wind down
        seen.append(progress.cancel_requested())
        return []

    monkeypatch.setattr(custom_sampler, "_drain", drain)


def test_cancel_reaches_the_workers_then_is_withdrawn(board, monkeypatch):
    seen = []
    _interrupted(monkeypatch, seen)

    with pytest.raises(comfy.model_management.InterruptProcessingException):
        nodes._ray_get_cancellable(["ref"])

    assert seen == [True]
    assert not progress.cancel_requested()


def test_withdrawn_even_when_the_drain_fails(board, monkeypatch):
    _interrupted(monkeypatch, [])

    def broken(futures, _checkpoint):
        raise RuntimeError("worker died")

    monkeypatch.setattr(custom_sampler, "_drain", broken)

    with pytest.raises(RuntimeError):
        nodes._ray_get_cancellable(["ref"])

    assert not progress.cancel_requested()
