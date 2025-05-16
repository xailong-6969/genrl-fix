import os
import pickle

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from genrl_swarm.communication.hivemind.hivemind_backend import HivemindBackend


def _test_hivemind_backend(rank, tmp_path, world_size):
    store = dist.FileStore(tmp_path, world_size)
    if rank == 0:
        backend = HivemindBackend(timeout=5, bootstrap=True)
        peers = pickle.dumps(backend.initial_peers)
        store.set("initial_peers", peers)
    else:
        peers = store.get("initial_peers")
        initial_peers = pickle.loads(peers)
        backend = HivemindBackend(initial_peers=initial_peers, timeout=5)
    obj = [rank]
    gathered_obj = backend.all_gather_object(obj)
    assert len(gathered_obj) == world_size
    gathered_obj = list(sorted(gathered_obj, key=lambda x: x[0]))
    assert gathered_obj == [[i] for i in range(world_size)]


@pytest.mark.parametrize("world_size", [1, 2])
def test_hivemind_backend(tmp_path, world_size):
    os.environ["WORLD_SIZE"] = str(world_size)
    mp.spawn(
        _test_hivemind_backend,
        args=(
            os.path.join(tmp_path, "file"),
            world_size,
        ),
        nprocs=world_size,
        join=True,
        daemon=False,
    )
