import json
import edn_format
try:
    import ujson as json_fast
except ImportError:
    json_fast = json
from edn_format.immutable_dict import ImmutableDict
from typing import Any, Set, Union


def parse_data(data: str, data_type: str) -> Any:
    """
    Parse JSON or EDN string into a Python data structure.

    :param data: The data string.
    :param data_type: 'json' or 'edn'.
    :return: Parsed data (dict, list, etc.).
    """
    if data_type == 'json':
        return json_fast.loads(data)
    elif data_type == 'edn':
        return edn_format.loads(data)
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")


def atomize(data: Any) -> Set[str]:
    """
    Atomize a data structure into its composite parts (keys and values).
    Recursively extracts all string keys and string/numeric values.
    Handles EDN-specific types like keywords, sets, symbols.

    :param data: Parsed data structure (dict, list, etc.).
    :return: Set of unique atoms (strings).
    """
    atoms = set()

    def _atomize_recursive(obj: Any):
        if isinstance(obj, (dict, ImmutableDict)):
            for key, value in obj.items():
                _atomize_recursive(key)
                _atomize_recursive(value)
        elif isinstance(obj, (list, tuple, frozenset, set)):
            for item in obj:
                _atomize_recursive(item)
        elif isinstance(obj, str):
            atoms.add(obj)
        elif isinstance(obj, (int, float)):
            atoms.add(str(obj))  # Convert numbers to strings for atomization
        elif isinstance(obj, edn_format.Keyword):
            atoms.add(f":{obj.name}")  # EDN keywords as :keyword
        elif isinstance(obj, edn_format.Symbol):
            atoms.add(obj.name)  # EDN symbols as strings
        elif isinstance(obj, edn_format.Char):
            atoms.add(str(obj))  # EDN characters
        elif obj is None:
            atoms.add("nil")
        elif isinstance(obj, bool):
            atoms.add(str(obj).lower())
        elif str(obj).lower() in ['true', 'false']:
            atoms.add(str(obj).lower())
        # Ignore other types

    _atomize_recursive(data)
    return atoms