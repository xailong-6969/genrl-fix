import abc
from typing import Any, Iterable, Sequence


class DataManager(abc.ABC):
    def initialize(self):
        """Optional initialization method."""
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
