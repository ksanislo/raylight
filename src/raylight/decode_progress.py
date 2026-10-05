"""Progress for the decodes, which run on several workers at once.

A decode is split across workers, so there is no single rank whose progress is
the whole job. Each piece of the work - a rank's share of the chunks, the stitch
on rank 0 - posts its own done/total to the progress board under its own key,
and the host sums them into one bar while it waits.

Every call swallows its errors: progress reporting must never break a decode.
"""
from raylight import progress

_PREFIX = "decode/"


def clear():
    """Forget the previous decode's progress. Called by the host before submitting."""
    progress.clear_keys(_PREFIX)


def write(part, done, total):
    """Record that `part` has finished `done` of its `total` units."""
    try:
        progress.put(_PREFIX + part, (int(done), int(total)))
    except (TypeError, ValueError):
        pass


def read():
    """Sum every part's progress, or None when nothing has been posted."""
    done = total = 0
    found = False
    for state in progress.items(_PREFIX).values():
        try:
            part_done, part_total = state
            done += int(part_done)
            total += int(part_total)
            found = True
        except (TypeError, ValueError):
            continue
    return (done, total) if found else None


class Relay:
    """Host side: show the summed progress on a ComfyUI progress bar."""

    def __init__(self):
        self._bar = None
        self._total = None

    def poll(self):
        state = read()
        if state is None or state[1] <= 0:
            return
        done, total = state
        if self._bar is None or total != self._total:
            import comfy.utils

            self._bar = comfy.utils.ProgressBar(total)
            self._total = total
        self._bar.update_absolute(min(done, total), total)
