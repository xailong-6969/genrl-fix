import pytest
import torch.multiprocessing as mp
import torch

from genrl_swarm.common.search import gather_prune_broadcast, all_gather_responses


def _test_gather_prune_broadcast(rank, init_method, world_size):
    torch.distributed.init_process_group(
        init_method=init_method, backend="gloo", rank=rank, world_size=world_size
    )
    rank = torch.distributed.get_rank()

    # (rounds, responses, stages, seqeunce)
    outputs = [
        [
            [[rank + 1], [rank + 2]],
        ],
        [
            [[rank + 1], [rank + 2]],
        ],
    ]
    _pruning_fn = lambda responses: torch.tensor(
        [-response[0][0] for response in responses], dtype=torch.float32
    )
    outputs = gather_prune_broadcast(
        outputs,
        _pruning_fn,
        beam_width=1,
        pg=None,
    )
    assert outputs == [
        [
            [[1], [2]],
        ],
        [
            [[1], [2]],
        ],
    ]
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_gather_prune_broadcast(tmp_path, world_size):
    init_method = f"file://{tmp_path}/shared_file"
    mp.spawn(
        _test_gather_prune_broadcast,
        args=(init_method, world_size),
        nprocs=world_size,
        join=True,
        daemon=True,
    )


def _test_all_gather_responses(rank, init_method, world_size):
    torch.distributed.init_process_group(
        init_method=init_method, backend="gloo", rank=rank, world_size=world_size
    )
    outputs = [
        [
            [[rank], [rank + 1]],
        ],
        [
            [[rank], [rank + 1]],
        ],
    ]
    outputs = all_gather_responses(outputs, pg=None)
    expected_outputs = [
        [[[irank], [irank + 1]] for irank in range(world_size)],
        [[[irank], [irank + 1]] for irank in range(world_size)],
    ]
    assert outputs == expected_outputs
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_all_gather_responses(tmp_path, world_size):
    init_method = f"file://{tmp_path}/shared_file"
    mp.spawn(
        _test_all_gather_responses,
        args=(init_method, world_size),
        nprocs=world_size,
        join=True,
        daemon=True,
    )
