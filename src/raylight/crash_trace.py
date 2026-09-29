"""Durable progress markers for a host that resets without logging anything.

The reset takes the machine instantly: no kernel message, no NMI, nothing in
the BMC log, and anything still in the page cache is gone with it. What
survives is whatever reached the disk, so every marker is fsynced before the
work it announces begins. The last line in a rank's file is then the last thing
that started, which is the only evidence available about where it died.

The pool here is ZFS with an Optane log device, so a sync costs tens of
microseconds. That is cheap enough to mark individual parameters and blocks
rather than phases: the whole shard load is ~1600 marks, about 25 ms, against a
load measured in tens of seconds. Resolution is worth far more than that.

Markers are written to real storage. Ray's tmpdir is tmpfs on this host, so a
marker written there would not survive the event it exists to record.

Every call swallows its errors: tracing must never be what breaks a render.
"""

import os
import threading
import time

_lock = threading.Lock()
_handle = None
_resolved = False
_start = time.monotonic()


def _directory():
    explicit = os.environ.get("RAYLIGHT_CRASH_TRACE_DIR")
    if explicit:
        return explicit
    #Workers are handed COMFYUI_BASE_DIRECTORY, whose parent is the install
    #root and is real storage.
    base = os.environ.get("COMFYUI_BASE_DIRECTORY")
    if base:
        return os.path.join(os.path.dirname(base.rstrip(os.sep)), "logs", "crash-trace")
    return None


def _open():
    global _handle, _resolved
    if _resolved:
        return _handle
    _resolved = True
    directory = _directory()
    if not directory:
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        _handle = open(os.path.join(directory, "pid-{}.log".format(os.getpid())), "a")
    except OSError:
        _handle = None
    return _handle


def mark(tag, rank=None):
    """Record that *tag* is starting, and do not return until it is on disk."""
    try:
        with _lock:
            handle = _open()
            if handle is None:
                return
            handle.write("{:9.3f} rank={} {}\n".format(
                time.monotonic() - _start, "?" if rank is None else rank, tag))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        pass
