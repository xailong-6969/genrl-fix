import logging
import os
from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Type
import hydra
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record


_LOG = logging.getLogger(__name__)


@dataclass
class _DistributedContext:
    backend: str

    def __enter__(self) -> None:
        torch.distributed.init_process_group(backend=self.backend)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        torch.distributed.destroy_process_group()


@record
def _main(cfg: DictConfig):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        backend = torch.distributed.Backend.NCCL
    else:
        backend = torch.distributed.Backend.GLOO

    log_dir = cfg.log_dir

    with _DistributedContext(backend):
        rank = torch.distributed.get_rank()
        if rank == 0:
            # Assume log_dir is in shared volume.
            os.makedirs(log_dir, exist_ok=True)
        torch.distributed.barrier()

        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{rank}.log")
        )
        _LOG.addHandler(file_handler)

        if rank == 0:
            _LOG.info(OmegaConf.to_yaml(cfg))
            _LOG.info(f"Using communication backend: {backend}.")
        torch.distributed.barrier()
        _LOG.info(
            f"Launching distributed training with {local_rank=} {rank=} {world_size=}."
        )
        # TODO(jkolehm): call game manager to start the actual training.


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    # Set error logging to error.log in the specified log directory.
    log_dir = cfg.log_dir
    os.environ["TORCHELASTIC_ERROR_FILE"] = os.path.join(log_dir, "error.log")
    _main(cfg)


if __name__ == "__main__":
    main()
