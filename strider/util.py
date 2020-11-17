"""General utilities."""
import logging
import logging.config
import re
from typing import Callable, Union

import httpx
import yaml

LOGGER = logging.getLogger(__name__)


async def post_json(url, request):
    """Make post request."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=request,
        )
        if response.status_code >= 300:
            raise KPError(
                "{}\n{}\n\x1b[31m{} Error:\x1b[0m {}".format(
                    url,
                    str(request),
                    response.status_code,
                    response.text,
                )
            )
        LOGGER.debug(
            "%s\n%s\n\x1b[32m%d Success:\x1b[0m %s",
            url,
            str(request),
            response.status_code,
            response.text,
        )
    return response.json()


class KPError(Exception):
    """Exception in a KP request."""


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


def snake_case(arg: Union[str, list[str]]):
    """Convert each string or set of strings to snake_case."""
    if isinstance(arg, str):
        return _snake_case(arg)
    elif isinstance(arg, list):
        try:
            return [snake_case(arg) for arg in arg]
        except AttributeError as err:
            raise ValueError from err
    else:
        raise ValueError()


def _spaced(arg: str):
    """Convert string to spaced format.

    _ is replaced with a space.
    """
    return re.sub('_', ' ', arg)


def spaced(arg: Union[str, list[str]]):
    """Convert each string or set of strings to spaced format."""
    if isinstance(arg, str):
        return _spaced(arg)
    elif isinstance(arg, list):
        try:
            return [spaced(arg) for arg in arg]
        except AttributeError as err:
            raise ValueError from err
    else:
        raise ValueError()


def setup_logging():
    """Set up logging."""
    with open('logging_setup.yml', 'r') as stream:
        config = yaml.load(stream.read(), Loader=yaml.SafeLoader)
    logging.config.dictConfig(config)


def ensure_list(arg: Union[str, list[str]]) -> list[str]:
    """Enclose in list if necessary."""
    if isinstance(arg, list):
        return arg
    return [arg]


def message_to_list_form(message):
    """Convert *graph nodes/edges and node/edge bindings to list forms."""
    if message['results']:
        message['results'] = [
            {
                'node_bindings': [
                    {
                        'qg_id': qg_id,
                        **binding,
                    }
                    for qg_id, binding in result['node_bindings'].items()
                ],
                'edge_bindings': [
                    {
                        'qg_id': qg_id,
                        **binding,
                    }
                    for qg_id, binding in result['edge_bindings'].items()
                ],
            } for result in message.get('results', [])
        ]
    if message['knowledge_graph']['nodes']:
        message['knowledge_graph']['nodes'] = [
            {
                'id': node['id'],
                **node,
            }
            for node in message['knowledge_graph']['nodes']
        ]
    if message['knowledge_graph']['edges']:
        message['knowledge_graph']['edges'] = [
            {
                'id': edge['id'],
                **edge,
            }
            for edge in message['knowledge_graph']['edges']
        ]
    return message


def message_to_dict_form(message):
    """Convert *graph nodes/edges and node/edge bindings to dict forms."""
    if message['results']:
        if isinstance(message['results'][0]['node_bindings'], list):
            message['results'] = [
                {
                    'node_bindings': {
                        binding['qg_id']: [binding]
                        for binding in result['node_bindings']
                    },
                    'edge_bindings': {
                        binding['qg_id']: [binding]
                        for binding in result['edge_bindings']
                    },
                } for result in message.get('results', [])
            ]
        elif not isinstance(
                list(message['results'][0]['node_bindings'].values())[0],
                dict
        ):
            message['results'] = [
                {
                    'node_bindings': {
                        key: [{'kg_id': el} for el in ensure_list(bindings)]
                        for key, bindings in result['node_bindings'].items()
                    },
                    'edge_bindings': {
                        key: [{'kg_id': el} for el in ensure_list(bindings)]
                        for key, bindings in result['edge_bindings'].items()
                    },
                } for result in message.get('results', [])
            ]
    if message['knowledge_graph']['nodes']:
        message['knowledge_graph']['nodes'] = {
            node['id']: node
            for node in message['knowledge_graph']['nodes']
        }
    if message['knowledge_graph']['edges']:
        message['knowledge_graph']['edges'] = {
            edge['id']: edge
            for edge in message['knowledge_graph']['edges']
        }
    return message


def graph_to_dict_form(graph):
    """Convert query_graph or knowledge_graph to dict form."""
    return {
        'nodes': {
            node['id']: node
            for node in graph['nodes']
        },
        'edges': {
            edge['id']: edge
            for edge in graph['edges']
        }
    }


def deduplicate_by(elements: list, fcn: Callable):
    """De-duplicate list via a function of each element."""
    return list(dict((fcn(element), element) for element in elements).values())
