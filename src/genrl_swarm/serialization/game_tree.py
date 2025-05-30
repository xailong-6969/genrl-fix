from enum import Enum
from typing import Any, Dict, List, Tuple, Type


class ObjType:
    LIST = 1
    DICT = 2
    STRING = 3


class Serializer:
    _SERIALIZATION_METHOD = {}
    _DESERIALIZATION_METHOD = {}

    @classmethod
    def from_bytes(cls, obj_type: ObjType):
        if obj_type not in cls._DESERIALIZATION_METHOD:
            raise RuntimeError(
                f"Unsupported type: {obj_type}; supported types are {list(cls._DESERIALIZATION_METHOD.keys())}."
            )
        return cls._DESERIALIZATION_METHOD[obj_type]

    @classmethod
    def to_bytes(cls, obj_type: Type):
        if obj_type not in cls._SERIALIZATION_METHOD:
            raise RuntimeError(
                f"Unsupported type: {obj_type}; supported types are {list(cls._SERIALIZATION_METHOD.keys())}."
            )
        return cls._SERIALIZATION_METHOD[obj_type]

    @classmethod
    def register_deserializer(cls, obj_type: ObjType):
        def wrapper(fn):
            cls._DESERIALIZATION_METHOD[obj_type] = fn
            return fn

        return wrapper

    @classmethod
    def register_serializer(cls, obj_type: Type):
        def wrapper(fn):
            cls._SERIALIZATION_METHOD[obj_type] = fn
            return fn

        return wrapper


def _from_bytes(b: bytes, i: int) -> Tuple[Any, int]:
    obj_type = int.from_bytes(b[i : (i + 8)], byteorder="big", signed=False)
    i += 8
    return Serializer.from_bytes(obj_type)(b, i)


@Serializer.register_deserializer(ObjType.STRING)
def string_from_bytes(b: bytes, i: int) -> Tuple[Dict[str, Any], int]:
    n_bytes = int.from_bytes(b[i : (i + 8)], byteorder="big", signed=False)
    i += 8
    s = b[i : (i + n_bytes)].decode("utf-8")
    i += n_bytes
    return s, i


@Serializer.register_deserializer(ObjType.LIST)
def list_from_bytes(b: bytes, i: int) -> Tuple[List[Any], int]:
    n_items = int.from_bytes(b[i : (i + 8)], byteorder="big", signed=False)
    i += 8
    out = [None] * n_items

    for k in range(n_items):
        out[k], i = _from_bytes(b, i)
    return out, i


@Serializer.register_deserializer(ObjType.DICT)
def dict_from_bytes(b: bytes, i: int) -> Tuple[List[Any], int]:
    n_items = int.from_bytes(b[i : (i + 8)], byteorder="big", signed=False)
    i += 8
    out = {}
    for _ in range(n_items):
        key, i = _from_bytes(b, i)
        value, i = _from_bytes(b, i)
        out[key] = value
    return out, i


@Serializer.register_serializer(str)
def string_to_bytes(obj: str) -> bytes:
    serialized_obj = obj.encode("utf-8")
    type_bytes = ObjType.STRING.to_bytes(length=8, byteorder="big", signed=False)
    len_bytes = len(serialized_obj).to_bytes(length=8, byteorder="big", signed=False)
    return type_bytes + len_bytes + serialized_obj


@Serializer.register_serializer(dict)
def dict_to_bytes(obj: Dict[Any, Any]) -> bytes:
    type_bytes = ObjType.DICT.to_bytes(length=8, byteorder="big", signed=False)
    len_bytes = len(obj).to_bytes(length=8, byteorder="big", signed=False)
    dict_bytes = []
    for key, value in obj.items():
        dict_bytes.append(to_bytes(key))
        dict_bytes.append(to_bytes(value))
    return type_bytes + len_bytes + b"".join(dict_bytes)


@Serializer.register_serializer(list)
def list_to_bytes(obj: List[Any]) -> bytes:
    type_bytes = ObjType.LIST.to_bytes(length=8, byteorder="big", signed=False)
    count_bytes = len(obj).to_bytes(length=8, byteorder="big", signed=False)
    header = type_bytes + count_bytes
    serialized_list = [to_bytes(x) for x in obj]
    return header + b"".join(serialized_list)


def to_bytes(obj: Any) -> bytes:
    return Serializer.to_bytes(type(obj))(obj)


def from_bytes(b: bytes) -> Any:
    return _from_bytes(b, 0)[0]
