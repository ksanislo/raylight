"""Place a worker on the NUMA node its GPU is attached to.

A worker that runs on one socket while its GPU hangs off the other pays the
interconnect twice: once for the host side of every transfer, and again for
NCCL's staging buffers. On a two socket box the remote access cost is roughly
double (node distance 21 against 10 locally), and it applies to the whole
model, which is streamed host to device on every step when dynamic VRAM is
offloading.

Affinity and memory policy are inherited by threads created afterwards, so this
has to run early in the worker's __init__, before the model is loaded.
"""

import ctypes
import logging
import os


def _gpu_sysfs_path(device_index=0):
    import torch

    props = torch.cuda.get_device_properties(device_index)
    return "/sys/bus/pci/devices/{:04x}:{:02x}:{:02x}.0".format(
        props.pci_domain_id, props.pci_bus_id, props.pci_device_id
    )


def gpu_numa_node(device_index=0):
    """The NUMA node a GPU is attached to, or None when unknown."""
    try:
        with open(os.path.join(_gpu_sysfs_path(device_index), "numa_node")) as fh:
            node = int(fh.read().strip())
    except Exception:
        return None
    return node if node >= 0 else None


def node_cpus(node):
    """The CPUs belonging to a NUMA node, parsed from its cpulist."""
    with open("/sys/devices/system/node/node{}/cpulist".format(node)) as fh:
        spec = fh.read().strip()
    cpus = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            cpus.update(range(int(first), int(last) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _prefer_node(node):
    """Prefer a node for future allocations. Preferred rather than bound: the
    local node may not have room for the model, and spilling is better than
    failing."""
    lib = ctypes.CDLL("libnuma.so.1")
    if lib.numa_available() < 0:
        return False
    lib.numa_set_preferred(ctypes.c_int(node))
    return True


def bind_to_gpu_node(device_index=0):
    """Pin this worker to its GPU's NUMA node. Returns a description of what
    was applied, or None when it was skipped or unavailable."""
    if os.environ.get("RAYLIGHT_NUMA_BIND") != "1":
        return None

    node = gpu_numa_node(device_index)
    if node is None:
        logging.info("[Raylight][NUMA] GPU reports no NUMA node, leaving placement alone")
        return None

    try:
        cpus = node_cpus(node)
    except Exception as e:
        logging.warning("[Raylight][NUMA] could not read cpulist for node %s: %s", node, e)
        return None
    if not cpus:
        return None

    try:
        os.sched_setaffinity(0, cpus)
    except Exception as e:
        logging.warning("[Raylight][NUMA] sched_setaffinity failed: %s", e)
        return None

    try:
        preferred = _prefer_node(node)
    except Exception as e:
        logging.warning("[Raylight][NUMA] memory policy unavailable: %s", e)
        preferred = False

    return "node={} cpus={} mem={}".format(
        node, len(cpus), "preferred" if preferred else "default"
    )
