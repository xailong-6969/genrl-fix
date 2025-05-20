import abc
from typing import Any, Iterable, Sequence


class DataManager(abc.ABC):
    def initialize(self):
        """Optional initialization method."""
        pass

    @abc.abstractmethod
    def get_round_data(self) -> Any: #TODO(discuss): Should we allow for a param/set of params for deterministic/consistent sampling for the round batch in here? Or should it be delegated to within game manager or somewhere else? (game manager might be only place to pass these details across the swarm)
        """Return a batch of data needed to define the start of a round."""
        pass

    @abc.abstractmethod
    def train_batch(self) -> Any:
        """Return single batch of training data."""
        pass

    @abc.abstractmethod
    def eval_data(self, name: str | None = None) -> Iterable[Any]:
        """Return iterable to eval data."""
        pass


class TokenizedDataManager(DataManager):
    @abc.abstractmethod
    def encode(self, text: str) -> Any:
        pass

    @abc.abstractmethod
    def decode(self, tokens: Any) -> str:
        pass
