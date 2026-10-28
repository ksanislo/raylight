"""A cancel left standing by an earlier prompt does not stop the next encode.

The cancel flag on the progress board is only reset by the host before it
dispatches, and a cancelled sampler leaves it set. The sharded encode runs before
the sampler in a prompt, so unless it resets the flag itself it stops at its first
block, and every later prompt with a new text stops there too.
"""
import pytest
import torch

if not torch.cuda.is_available():
    # comfy picks its device at import and needs telling there is no card
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

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


class _Method:
    def __init__(self, name, seen):
        self.name = name
        self.seen = seen

    def remote(self, *args):
        # what a worker would read as it starts this call
        self.seen.append((self.name, progress.cancel_requested(force=True)))
        return self.name


class _Worker:
    def __init__(self, seen):
        self.prepare_clip = _Method("prepare_clip", seen)
        self.encode_tokens = _Method("encode_tokens", seen)


def test_encode_resets_a_cancel_left_by_an_earlier_prompt(board, monkeypatch):
    seen = []
    workers = [_Worker(seen), _Worker(seen)]

    def gather(futures, _ray_actors):
        return [{"cond": torch.zeros(1)} for _ in futures]

    monkeypatch.setattr(custom_sampler, "_gather_with_progress", gather)

    # a sampler was cancelled and its workers wound down
    progress.request_cancel()

    clip = nodes.RayCLIP({"workers": workers}, None, "te.safetensors", "minimax")
    clip.encode_from_tokens({"l": []})

    assert seen and all(not cancelled for _name, cancelled in seen)
    assert not progress.cancel_requested(force=True)
