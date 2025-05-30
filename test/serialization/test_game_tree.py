import pytest

from genrl_swarm.serialization.game_tree import from_bytes, to_bytes


@pytest.mark.parametrize(
    "obj",
    [
        {"A": ["this", "is", "a cat"], "B": ["this", "is", "a cat", "and dog"]},
        {
            "A": {"there is data": "here"},
            "B": {"this": {"is": {"nested": "dictionary"}}},
        },
    ],
)
def test_to_and_from_bytes(obj):
    serialized_obj = to_bytes(obj)
    deserialized_obj = from_bytes(serialized_obj)
    assert deserialized_obj == obj
