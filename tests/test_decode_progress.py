import pytest

import raylight.decode_progress as decode_progress


@pytest.fixture(autouse=True)
def channel(tmp_path, monkeypatch):
    monkeypatch.setenv("RAYLIGHT_RAY_TMPDIR", str(tmp_path))
    return tmp_path


def test_parts_sum_into_one_figure():
    decode_progress.write("video_rank0", 2, 6)
    decode_progress.write("video_rank1", 3, 5)
    decode_progress.write("video_stitch", 0, 11)

    assert decode_progress.read() == (5, 22)


def test_nothing_written_reads_as_none():
    assert decode_progress.read() is None


def test_clear_forgets_the_previous_decode(channel):
    decode_progress.write("video_rank0", 6, 6)
    (channel / "unrelated").write_text("kept")

    decode_progress.clear()

    assert decode_progress.read() is None
    assert (channel / "unrelated").exists()


def test_a_half_written_part_is_skipped(channel):
    decode_progress.write("video_rank0", 1, 4)
    (channel / "decode_progress_video_rank1").write_text("{not json")

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
