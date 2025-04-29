from dataclasses import dataclass
from typing import Any, List
from torch import Tensor
from numpy import ndarray
@dataclass
class GameState:
    round: int
    stage: int
    round_data: Any
    outputs: List[List[List[List[Any]]]] | None # [Agent][Batch][Stage][Generation]
    swarm_size: int=1 # Number of agents

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
        for i, output in enumerate(outputs):
            self.outputs[0][i][-1].append(output) # TODO: agent-0, batch-i, last stage, should we track our agent-id or assume idx 0?

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
