import pytest

import raylight.decode_progress as decode_progress
from raylight import progress


@pytest.fixture(autouse=True)
def board():
    board = progress.Board()
    progress.use_local_board(board)
    yield board
    progress.use_local_board(None)


def test_parts_sum_into_one_figure():
    decode_progress.write("video_rank0", 2, 6)
    decode_progress.write("video_rank1", 3, 5)
    decode_progress.write("video_stitch", 0, 11)

    assert decode_progress.read() == (5, 22)


def test_nothing_written_reads_as_none():
    assert decode_progress.read() is None


def test_clear_forgets_the_previous_decode(board):
    decode_progress.write("video_rank0", 6, 6)
    board.put("sampler_progress", (1, 2, 0))

    decode_progress.clear()

    assert decode_progress.read() is None
    assert board.get("sampler_progress") == (1, 2, 0)


def test_a_malformed_part_is_skipped(board):
    decode_progress.write("video_rank0", 1, 4)
    board.put("decode/video_rank1", "not a pair")

    assert decode_progress.read() == (1, 4)


def test_the_relay_drives_a_progress_bar(monkeypatch):
    bars = []

    class Bar:
        def __init__(self, total):
            self.total, self.updates = total, []
            bars.append(self)

        def update_absolute(self, value, total=None):
            self.updates.append((value, total))

    import comfy.utils

    monkeypatch.setattr(comfy.utils, "ProgressBar", Bar)
    relay = decode_progress.Relay()
    relay.poll()
    assert bars == []

    decode_progress.write("video_rank0", 1, 3)
    relay.poll()
    decode_progress.write("video_rank0", 3, 3)
    relay.poll()
    assert len(bars) == 1 and bars[0].updates == [(1, 3), (3, 3)]

    # a part joining late changes the total, and the bar is rebuilt for it
    decode_progress.write("video_stitch", 0, 3)
    relay.poll()
    assert len(bars) == 2 and bars[1].total == 6
