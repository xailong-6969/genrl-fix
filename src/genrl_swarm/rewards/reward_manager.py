import abc
from typing import Any, Callable, Union, Iterable, Dict
from genrl_swarm.rewards.reward_store import RewardFnStore
# from genrl_swarm.game_state import GameState todo: add game state
GameState = Any


class RewardManager(abc.ABC):
    @abc.abstractmethod
    def update_rewards(self, game_state: GameState) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self) -> Any:
        pass


class DefaultRewardManager(RewardManager):
    def __init__(self, reward_fn_store: RewardFnStore):
        self._round = 0
        self._stage = 0
        self._rewards = None
        self.reward_fn_store = reward_fn_store

    @property
    def round(self) -> int:
        return self._round

    @round.setter
    def round(self, value: int) -> None:
        if value < 0:
            value = 0
        self._round = value

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        if value < 0:
            value = 0
        self._stage = value

    @property
    def rewards(self) -> Union[Iterable, Dict]:
        return self._rewards

    @rewards.setter
    def rewards(self, value: Union[Iterable, Dict]) -> None:
        self._rewards = value

    def set_round_stage(self, round: int, stage: int) -> None:
        self.round = round
        self.stage = stage

    def dispatch_reward_fn(self, round: int, stage: int) -> Callable:
        return self.reward_fn_store[round][stage]

    def __call__(self, round: int, stage: int, game_state: GameState) -> Union[Iterable, Dict]:
        """
        Dispatch the reward function for the given round and stage and return the rewards.
        Side Effects: Sets the rewards attribute.
        """
        reward_fn = self.dispatch_reward_fn(round, stage)
        rewards = reward_fn(game_state)
        self.rewards = rewards
        return rewards

    def reset(self) -> None:
        self._stage = 0
        self._rewards = None

    def update_rewards(self, game_state: GameState) -> None:
        self.__call__(game_state.round, game_state.stage, game_state)
