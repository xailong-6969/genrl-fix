from typing import Any, Callable
from itertools import chain, islice
import torch
import torch.distributed as dist


def _prune(
    responses: list[list[Any]],
    pruning_fn: Callable | None,
    beam_width: int,
) -> list[list[Any]]:
    if beam_width <= 0:
        return responses
    scores = pruning_fn(responses)
    _, topk_indices = torch.topk(scores, beam_width)
    return [responses[index.item()] for index in topk_indices]


def _gather_responses(outputs: list[list[list[Any]]], pg: dist.ProcessGroup):
    rank = dist.get_rank()
    root = dist.get_global_rank(pg, 0) if pg else 0
    group_size = dist.get_world_size(group=pg)
    buffer_ = [None for _ in range(group_size)]
    num_rounds = len(outputs)

    if rank == root:
        # Gather all partial decoding results from the
        dist.gather_object(outputs, buffer_, dst=root, group=pg)

        # Combine responses (round, responses, stage, response)
        return [
            list(chain(*[buffer_[irank][r] for irank in range(group_size)]))
            for r in range(num_rounds)
        ]
    else:
        dist.gather_object(outputs, dst=root, group=pg)
        return []


def _broadcast_responses(
    all_responses: list[list[list[Any]]] | None, pg: dist.ProcessGroup
) -> list[list[list[Any]]]:
    root = dist.get_global_rank(pg, 0) if pg else 0
    buffer_ = [all_responses]
    dist.broadcast_object_list(
        buffer_,
        src=root,
        group=pg,
    )
    return buffer_[0]


def gather_prune_broadcast(
    outputs: list[list[list[Any]]],
    pruning_fn: Callable | None,
    beam_width: int,
    pg: dist.ProcessGroup,
) -> list[list[list[Any]]]:
    rank = dist.get_rank()
    root = dist.get_global_rank(pg, 0) if pg else 0

    # Gather all outputs
    all_outputs = _gather_responses(outputs, pg)

    if rank == root:
        # Prune outputs to match the specified beam width.
        all_outputs = [
            _prune(round_outputs, pruning_fn, beam_width)
            for round_outputs in all_outputs
        ]
    return _broadcast_responses(all_outputs, pg)


def all_gather_responses(
    responses: list[list[list[Any]]],
    pg: dist.ProcessGroup,
) -> list[list[list[Any]]]:
    num_rounds = len(responses)
    group_size = dist.get_world_size(group=pg)
    buffer_ = [None for _ in range(group_size)]
    dist.all_gather_object(buffer_, responses, group=pg)
    return [
        list(chain(*[buffer_[irank][r] for irank in range(group_size)]))
        for r in range(num_rounds)
    ]
