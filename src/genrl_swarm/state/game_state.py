from dataclasses import dataclass
from typing import Any, List, Callable, Optional
from torch import Tensor
from numpy import ndarray


@dataclass
class GameState:
    round: int
    stage: int
    round_data: Any
    outputs: List[List[List[List[Any]]]] | None # [Agent][Batch][Stage][Generation]
    swarm_size: int=1 # Number of agents
    rank: int = 0  # Current rank in distributed setup

    def __post_init__(self) -> None:
        if self.round < 0:
            self.round = 0
        if self.stage < 0:
            self.stage = 0

        if self.round_data is not None:
            self.batch_size = len(self.round_data)
        else:
            self.batch_size = 0

        if self.outputs is None:
            self.outputs = self._init_agent_batch_stage_grid()

    def _init_agent_batch_stage_grid(self):
        grid = []
        for _ in range(self.swarm_size):
            agent_list = []
            for _ in range(self.batch_size):
                batch_list = [[]] # each batch tracks its own stage list
                agent_list.append(batch_list)
            grid.append(agent_list)
        return grid

    def append_generation(self, outputs: Any) -> None:
        if isinstance(outputs, (Tensor, ndarray)):
            outputs = outputs.tolist()

        for i, agent_responses in enumerate(outputs):
            for j, batch in enumerate(agent_responses):
                for generation in batch:
                    self.outputs[i][j][-1].append(generation)

    def get_latest(self) -> Any:
        """Returns the generations from the latest stage.
        
        Returns:
            Any: The latest generated outputs from the current stage.
        """
        # Get the last generation from each batch for the current agent (assumed to be 0)
        if self.stage > 0 and self.outputs and len(self.outputs) > 0:
            # Access the last element of the previous stage (current stage - 1)
            stage_idx = self.stage - 1
            latest_outputs = []
            for agent_idx in range(self.swarm_size):
                agent_batches = []
                for batch_idx in range(self.batch_size):
                    batch_generations = []
                    if len(self.outputs[agent_idx][batch_idx]) > stage_idx and self.outputs[agent_idx][batch_idx][stage_idx]:
                        batch_generations.extend(self.outputs[agent_idx][batch_idx][stage_idx])
                    agent_batches.append(batch_generations)
                latest_outputs.append(agent_batches)
            return latest_outputs
        return None

    def advance_stage(self) -> None:
        self.stage += 1
        for agent in range(self.swarm_size):
            for batch in range(self.batch_size):
                self.outputs[agent][batch].append([])

    def advance_round(self, round_data: Any) -> None:
        self.round += 1
        self.stage = 0
        self.round_data = round_data
        self.batch_size = len(round_data)
        self.outputs = self._init_agent_batch_stage_grid()