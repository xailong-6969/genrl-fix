import abc
from typing import List, Any
from genrl_swarm.rewards import RewardManager
# from genrl_swarm.data_manager import DataManager todo: add data manager
# from genrl_swarm.game_state import GameState todo: add game state
DataManager = Any
GameState = Any

#TODO: Update to mirror discussions --> Should be able to directly accept inputs from state in the predefined format and then tokenize+generate+etc. and then produce output in the predefined format so state is able to parse/append/work with output
#NOTE: Predefined format is Dict[List[List[Tuple[Any]]]] where indices correspond to the following [Agents][Batch][Node Idx in Stage][World State] 
#NOTE: For output, probably don't need that final dimension since actions/outputs are treated as a singleton item in game tree nodes (regardless of how "complex" the datastructure a single model's rollout is)
class TrainerModule(abc.ABC):
    @abc.abstractmethod
    def __init__(self, models: List[Any], **kwargs):
        pass

    @abc.abstractmethod
    def generate(self, inputs: Any) -> Any:
        pass
    
    @abc.abstractmethod
    def train(self, game_state: GameState, reward_manager: RewardManager) -> None:
        pass
    
    @abc.abstractmethod
    def evaluate(self, data_manager: DataManager, reward_manager: RewardManager) -> None:
        pass

    @abc.abstractmethod
    def save(self, save_dir: str) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, load_dir: str) -> 'TrainerModule':
        pass
    

    #NOTE: Probably scrap this, but tossing in here since we probably want to provide some utility functions for mapping from tree indexing as well as from (mentioned in comment above)
    # def _map_to_tree_indexing(self, actions: Any, stage_mapping: dict) -> dict:
    #     """Maps actions with shape (N, ...) to expected format with shape (self.batch_size, num_nodes_in_stage, ...)"""
    #     batch = {}
    #     for action_idx in range(len(actions)):
    #         batch_idx, node_idx = stage_mapping[action_idx]
    #         if batch_idx not in batch:
    #             batch[batch_idx] = {node_idx: actions[action_idx]}
    #         else:
    #             batch[batch_idx][node_idx] = actions[action_idx]
    #     return batch