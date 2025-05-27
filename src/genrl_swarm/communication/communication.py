import abc
from typing import Any, Sequence


class Communication(abc.ABC):
    @abc.abstractmethod
    def all_gather_object(self, obj: Any, *args, **kwargs) -> Sequence[Any]:
        pass

    @abc.abstractmethod
    def get_id(self) -> int | str:
        pass