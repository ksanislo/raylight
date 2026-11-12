"""pin_ranks_to_gpus starts rank i on the i-th GPU of GPU_SELECT and Ray accounts for every GPU."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")

# Runs in its own interpreter: a local Ray cluster with four GPUs that do not exist.
# Ray still assigns their ids and sets CUDA_VISIBLE_DEVICES, which is all pinning uses.
SCRIPT = r'''
import os, sys, types
sys.path.insert(0, sys.argv[1])
order = [int(g) for g in sys.argv[2].split(",")]
import ray
from raylight.distributed_worker import ray_worker
from raylight.distributed_worker.raylightworker_config_schema import RayWorkerConfig


class Worker:
    def __init__(self, local_rank, device_id, config, world_size, group_id=0):
        self.local_rank = local_rank

    def placement(self):
        return self.local_rank, os.environ.get("CUDA_VISIBLE_DEVICES")


ray_worker.RayWorker = Worker
ray_worker._note_workers_spawned = lambda names: list(names)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in sorted(order))
ray.init(num_cpus=8, num_gpus=len(order), include_dashboard=False, log_to_driver=False)
config = types.SimpleNamespace(worker=RayWorkerConfig(selected_gpus=tuple(order), pin_ranks_to_gpus=True))
spawn = ray_worker.make_ray_actor_fn(len(order), config)
for attempt in range(2):  # a respawn releases the first reservation
    actors = spawn()["workers"]
    placed = ray.get([a.placement.remote() for a in actors])
    print("PLACED", attempt, placed, flush=True)
    assert placed == [(rank, str(gpu)) for rank, gpu in enumerate(order)], placed
    # every GPU is held through Ray: nothing is left for another GPU task
    assert ray.available_resources().get("GPU", 0) == 0, ray.available_resources()
    for a in actors:
        ray.kill(a, no_restart=True)
ray.shutdown()
print("OK")
'''


@pytest.mark.parametrize("order", ["2,0,3,1", "0,1,2,3", "3,2,1,0"])
def test_ranks_follow_gpu_select(order):
    result = subprocess.run([sys.executable, "-c", SCRIPT, SRC, order], capture_output=True, text=True,
                            timeout=300, env={**os.environ, "RAY_DEDUP_LOGS": "0"})
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout[-3000:] + result.stderr[-3000:]
