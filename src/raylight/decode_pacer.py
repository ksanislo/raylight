"""Start each rank's decode only once the card ahead of it is genuinely working.

Every rank wants the same weights from the same file, and the file is on spinning
disks. The first rank to ask starts the read and the others queue behind it on
the same blocks, so when it completes they are all released together - a gap
arranged beforehand does not survive that, and a longer one only means they pile
up more completely. Worse, the decode pulses, so four cards that begin together
stay together: if they are not separated well enough to start with, the load
cycling pulls them back into step on its own.

So nothing is timed. A rank waits until the card ahead of it has reached full
load, which cannot be true until that rank has faulted in what it needs - which
in turn leaves those pages cached for everyone behind it. The first card pays the
disk alone while the rest wait, and each one after it starts from memory. The
separation then holds without depending on the cache being warm beforehand,
because the gate re-establishes that condition itself on every decode.

Which peg counts is decided per request. The cards run at full load through
sampling and again for the audio decode, so "this card has reached full load" is
already true of all of them before a decode begins; only a peg seen after the
rank ahead was admitted says anything about whether it has started.

Full load is the mark rather than something lower because a card still pulling
weights sits far below it - measured, loading holds ten to twenty percent and
working reaches eighty and above, with nothing in between.

NVML reports every card whatever CUDA_VISIBLE_DEVICES says and creates no
context. Sampling runs in a thread so an answer never waits on the driver, and
waiting is left to the caller: Ray serialises calls into an actor, so sleeping
here would queue every other rank behind the one being held back.
"""

import ctypes
import threading
import time

import ray


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


@ray.remote(num_cpus=0)
class DecodePacer:
    def __init__(self, hold_seconds=0.7, busy_percent=80, poll_hz=20.0, max_wait=None,
                 max_together=2):
        self._hold = float(hold_seconds)
        self._busy = int(busy_percent)
        #A rank that never sees the one ahead get going is let through rather than
        #stalled, but the bound has to be short: a peg can be missed outright if
        #the work is briefer than a sample, and three ranks each waiting a minute
        #would cost far more than the coincidence being avoided.
        self._max_wait = float(max_wait) if max_wait else self._hold * 4.0
        #Two cards falling back into step is common and untroubled - measured at
        #68% of a decode. Three is where it gets close to a problem, so that is
        #what is worth delaying for, rather than every pair.
        self._max_together = max(1, int(max_together))
        self._interval = 1.0 / float(poll_hz)

        self._nvml = ctypes.CDLL("libnvidia-ml.so.1")
        self._nvml.nvmlInit_v2()
        count = ctypes.c_uint()
        self._nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
        self._handles = []
        for i in range(count.value):
            handle = ctypes.c_void_p()
            self._nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle))
            self._handles.append(handle)

        self._last_peg = [None] * len(self._handles)
        self._last_start = {}
        self._device_of = {}
        self._admitted_at = {}
        self._peg_after_admit = {}

        self._granted = 0
        self._held = 0
        self._forced = 0

        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            now = time.monotonic()
            for i, handle in enumerate(self._handles):
                sample = _Util()
                if self._nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(sample)) != 0:
                    continue
                if sample.gpu >= self._busy:
                    self._last_peg[i] = now
            time.sleep(self._interval)

    def _ahead_is_working(self, position, device, now):
        """True once the rank at *position* reached full load after it was let in."""
        admitted = self._admitted_at.get(position)
        peg = self._last_peg[device]
        if admitted is None or peg is None or peg < admitted:
            return False
        first = self._peg_after_admit.get(position)
        if first is None:
            first = self._peg_after_admit[position] = peg
        return (now - first) >= self._hold

    def _admit(self, key, device, now):
        self._admitted_at.setdefault(key, now)
        self._last_start[device] = now
        self._granted += 1
        return 0.0

    def acquire(self, position, device, round_index=0, waited=0.0):
        """Return 0 to start now, or how long to wait before asking again."""
        now = time.monotonic()
        self._device_of[position] = device
        key = (position, round_index)

        if position == 0:
            return self._admit(key, device, now)

        if round_index == 0:
            #The first chunk is the one that has to be chained. Every rank wants
            #the same weights off the same disks, so without this they queue on
            #one read and the kernel hands them all back together.
            ahead = self._device_of.get(position - 1)
            if ahead is not None and self._ahead_is_working((position - 1, 0), ahead, now):
                return self._admit(key, device, now)
        else:
            #Afterwards the spread is already there and only needs defending, so
            #a rank is held solely when enough others have just started that it
            #would be joining a crowd.
            crowd = sum(1 for other, started in self._last_start.items()
                        if other != device and (now - started) < self._hold * 2.0)
            if crowd < self._max_together:
                return self._admit(key, device, now)

        if waited >= self._max_wait:
            #A rank whose predecessor never gets going is let through rather than
            #stalled: finishing late is worse than two cards briefly coinciding.
            self._forced += 1
            return self._admit(key, device, now)

        self._held += 1
        return self._interval * 2

    def stats(self):
        return {"granted": self._granted, "held": self._held, "forced": self._forced,
                "hold": self._hold, "busy_percent": self._busy}

    def shutdown(self):
        self._running = False
