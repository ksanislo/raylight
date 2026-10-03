"""Progress for the decodes, which run on several workers at once.

A decode is split across workers, so there is no single rank whose progress is
the whole job. Each piece of the work - a rank's share of the chunks, the stitch
on rank 0 - writes its own done/total to a small file, and the host sums every
file into one bar while it waits. Like the sampler's side channel this relies on
host and workers sharing a filesystem; elsewhere nothing is found and the bar is
simply not shown.

Every call swallows its errors: progress reporting must never break a decode.
"""
import json
import os
import tempfile

_PREFIX = "decode_progress_"


def _dir():
    return (os.environ.get("RAYLIGHT_RAY_TMPDIR")
            or os.environ.get("RAY_TMPDIR")
            or os.path.join(tempfile.gettempdir(), "raylight-ray"))


def clear():
    """Forget the previous decode's progress. Called by the host before submitting."""
    try:
        for name in os.listdir(_dir()):
            if name.startswith(_PREFIX):
                try:
                    os.unlink(os.path.join(_dir(), name))
                except OSError:
                    pass
    except OSError:
        pass


def write(part, done, total):
    """Record that `part` has finished `done` of its `total` units."""
    try:
        os.makedirs(_dir(), exist_ok=True)
        target = os.path.join(_dir(), _PREFIX + part)
        tmp = target + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"done": int(done), "total": int(total)}, f)
        os.replace(tmp, target)
    except OSError:
        pass


def read():
    """Sum every part's progress, or None when nothing has been written."""
    done = total = 0
    found = False
    try:
        names = [n for n in os.listdir(_dir()) if n.startswith(_PREFIX) and not n.endswith(".tmp")]
    except OSError:
        return None
    for name in names:
        try:
            with open(os.path.join(_dir(), name)) as f:
                state = json.load(f)
            done += int(state["done"])
            total += int(state["total"])
            found = True
        except (OSError, ValueError, KeyError, TypeError):
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
