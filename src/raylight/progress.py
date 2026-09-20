"""Side channel carrying sampler progress from Ray workers back to the host.

Ray actors are single threaded, so while rank 0 is sampling it cannot answer a
status call. Rank 0 writes its progress to small files in Raylight's temp
directory and the host reads them while it waits on the sampling futures.
Host and workers share a filesystem on a local cluster; elsewhere the reads
find nothing and the UI simply behaves as it did before.

Every call swallows its errors: progress reporting must never break a render.
"""

import json
import os
import tempfile
import time

_STATE = "sampler_progress.json"
_PREVIEW = "sampler_preview.jpg"
_MIN_INTERVAL = 0.25

_last_write = 0.0


def _dir():
    base = (os.environ.get("RAYLIGHT_RAY_TMPDIR")
            or os.environ.get("RAY_TMPDIR")
            or os.path.join(tempfile.gettempdir(), "raylight-ray"))
    return base


def _path(name):
    return os.path.join(_dir(), name)


def clear():
    for name in (_STATE, _PREVIEW):
        try:
            os.unlink(_path(name))
        except OSError:
            pass


def write(value, total, force=False, preview_seq=0):
    """Publish a progress fraction. Throttled unless force is set.

    preview_seq lets the host tell whether a new preview image accompanies
    this update, so it only pays to decode one when there is something new.
    """
    global _last_write
    now = time.monotonic()
    if not force and now - _last_write < _MIN_INTERVAL:
        return
    _last_write = now
    try:
        target = _path(_STATE)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w") as handle:
            json.dump({"value": int(value), "total": int(total),
                       "preview_seq": int(preview_seq)}, handle)
        os.replace(tmp, target)
    except (OSError, ValueError):
        pass


def read():
    """Return (value, total, preview_seq) or None."""
    try:
        with open(_path(_STATE)) as handle:
            data = json.load(handle)
        return int(data["value"]), int(data["total"]), int(data.get("preview_seq", 0))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_preview(image, image_format="JPEG"):
    """Store a preview image produced by ComfyUI's previewer."""
    try:
        target = _path(_PREVIEW)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        image.save(tmp, format=image_format, quality=85)
        os.replace(tmp, target)
    except Exception:
        pass


def read_preview():
    try:
        from PIL import Image
        path = _path(_PREVIEW)
        with open(path, "rb") as handle:
            data = handle.read()
        import io
        return Image.open(io.BytesIO(data)).copy()
    except Exception:
        return None
