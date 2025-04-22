import abc
from typing import List, Any
# from genrl_swarm.reward_manager import RewardManager todo: add reward manager
# from genrl_swarm.data_manager import DataManager todo: add data manager
# from genrl_swarm.game_state import GameState todo: add game state
RewardManager = Any
DataManager = Any
GameState = Any

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
    