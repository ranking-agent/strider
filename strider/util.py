"""General utilities."""
import logging
import logging.config
import re
import typing

import yaml


def _snake_case(arg: str):
    """Convert string to snake_case.

    Non-alphanumeric characters are replaced with _.
    CamelCase is replaced with snake_case.
    """
    # replace non-alphanumeric characters with _
    tmp = re.sub(r'\W', '_', arg)
    # replace X with _x
    tmp = re.sub(
        r'(?<=[a-z])[A-Z](?=[a-z])',
        lambda c: '_' + c.group(0).lower(),
        tmp
    )
    # lower-case first character
    tmp = re.sub(
        r'^[A-Z](?=[a-z])',
        lambda c: c.group(0).lower(), 
        tmp
    )
    return tmp


def snake_case(arg: typing.Union[str, typing.List[str]]):
    """Convert each string or set of strings to snake_case."""
    if isinstance(arg, str):
        return _snake_case(arg)
    elif isinstance(arg, list):
        try:
            return [snake_case(arg) for arg in arg]
        except AttributeError:
            raise ValueError()
    else:
        raise ValueError()


def _spaced(arg: str):
    """Convert string to spaced format.

    _ is replaced with a space.
    """
    return re.sub('_', ' ', arg)


def spaced(arg: typing.Union[str, typing.List[str]]):
    """Convert each string or set of strings to spaced format."""
    if isinstance(arg, str):
        return _spaced(arg)
    elif isinstance(arg, list):
        try:
            return [spaced(arg) for arg in arg]
        except AttributeError:
            raise ValueError()
    else:
        raise ValueError()


def setup_logging():
    """Set up logging."""
    with open('logging_setup.yml', 'r') as stream:
        config = yaml.load(stream.read(), Loader=yaml.SafeLoader)
    logging.config.dictConfig(config)


def ensure_list(arg):
    """Enclose in list if necessary."""
    if isinstance(arg, list):
        return arg
    return [arg]
