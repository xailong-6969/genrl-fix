import abc
from typing import Any, Dict


class Communication(abc.ABC):
    @abc.abstractmethod
    def all_gather_object(self, obj: Any, *args, **kwargs) -> Dict[str | int, Any]:
        pass

    @abc.abstractmethod
    def get_id(self) -> int | str:
        pass