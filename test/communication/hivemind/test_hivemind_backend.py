import os

import pytest
import torch.multiprocessing as mp

from genrl_swarm.communication.hivemind.hivemind_backend import (
    HivemindBackend, HivemindRendezvouz)


def _test_hivemind_backend(rank, world_size):
    HivemindRendezvouz.init(is_master=rank == 0)

    backend = HivemindBackend(timeout=5)
    obj = [rank]
    gathered_obj = backend.all_gather_object(obj)
    assert len(gathered_obj) == world_size
    gathered_obj = list(sorted(gathered_obj, key=lambda x: x[0]))
    assert gathered_obj == [[i] for i in range(world_size)]


@pytest.mark.parametrize("world_size", [1, 2])
def test_hivemind_backend(world_size):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29400"
    mp.spawn(
        _test_hivemind_backend,
        args=(world_size,),
        nprocs=world_size,
        join=True,
        daemon=False,
    )
