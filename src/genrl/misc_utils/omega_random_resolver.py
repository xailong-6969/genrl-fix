import random

from omegaconf import ListConfig, OmegaConf


def random_choice_resolver(list_input):
    """
    OmegaConf resolver to randomly select an element from a list.
    Accepts:
    1. An OmegaConf ListConfig object (if passed directly, though usually it's a string).
    2. A Python list (if you somehow pass it this way, unlikely from config).
    """
    items = []
    if isinstance(list_input, (ListConfig, list)):
        items = list_input
    else:
        raise TypeError(
            f"random_choice_resolver expects a list-like input (ListConfig, list), got: {type(list_input)}"
        )

    if not items:
        return None
    return random.choice(items)


# Register the resolver with OmegaConf
OmegaConf.register_new_resolver("random_choice", random_choice_resolver)
