"""General utilities."""
import functools
import re
from typing import Callable, Union

import httpx
from starlette.middleware.cors import CORSMiddleware
import yaml
from bmt import Toolkit as BMToolkit


def camel_to_snake(s, sep=' '):
    return re.sub(r'(?<!^)(?=[A-Z])', sep, s).lower()


def snake_to_camel(s):
    return ''.join(word.title() for word in s.split(' '))


def function_to_mapping(f):
    """
    Given a function, generate an instance of a class that
    implements the __getitem__ interface
    """
    class Mapping():
        def __getitem__(self, lookup):
            value = f(lookup)
            if value is None:
                raise KeyError
            else:
                return value

        def __contains__(self, lookup):
            return f(lookup) is not None

    return Mapping()


class WrappedBMT():
    """
    Wrapping around some of the BMT Toolkit functions
    to provide case conversions to the new format
    """

    def __init__(self):
        self.bmt = BMToolkit(
            schema="https://raw.githubusercontent.com/biolink/biolink-model/1.8.0/biolink-model.yaml")
        self.all_slots = self.bmt.get_all_slots()
        self.all_slots_formatted = ['biolink:' + s.replace(' ', '_')
                                    for s in self.all_slots]
        self.prefix = 'biolink:'

        self.entity_prefix_mapping = function_to_mapping(self.entity_prefixes)

    def new_case_to_old_case(self, s):
        """
        Convert new biolink case format (biolink:GeneOrGeneProduct)
        to old case format (gene or gene product)

        Also works with slots (biolink:related_to -> related to)
        """
        s = s.replace(self.prefix, '')
        if s in self.all_slots_formatted:
            return s.replace('_', ' ')
        else:
            return camel_to_snake(s)

    def old_case_to_new_case(self, s):
        """
        Convert old case format (gene or gene product)
        to new biolink case format (biolink:GeneOrGeneProduct)

        Also works with slots (related to -> biolink:related_to)
        """
        if s in self.all_slots:
            return self.prefix + s.replace(' ', '_')
        else:
            return self.prefix + snake_to_camel(s)

    def get_descendants(self, concept):
        """ Wrapped BMT descendants function that does case conversions """
        concept_old_format = self.new_case_to_old_case(concept)
        descendants_old_format = self.bmt.get_descendants(concept_old_format)
        descendants = [self.old_case_to_new_case(d)
                       for d in descendants_old_format]
        if len(descendants) == 0:
            descendants.append(concept)
        return descendants

    def get_ancestors(self, concept):
        """ Wrapped BMT ancestors function that does case conversions """
        concept_old_format = self.new_case_to_old_case(concept)
        ancestors_old_format = self.bmt.get_ancestors(concept_old_format)
        ancestors = [self.old_case_to_new_case(a)
                     for a in ancestors_old_format]
        if len(ancestors) == 0:
            ancestors.append(concept)
        return ancestors

    def predicate_is_symmetric(self, predicate):
        """ Get whether a given predicate is symmetric """
        predicate_old_format = self.new_case_to_old_case(predicate)
        predicate_element = self.bmt.get_element(predicate_old_format)
        if not predicate_element:
            # Not in the biolink model
            return False
        return predicate_element.symmetric

    def predicate_inverse(self, predicate):
        """ Get the inverse of a predicate if it has one """
        predicate_old_format = self.new_case_to_old_case(predicate)
        predicate_element = self.bmt.get_element(predicate_old_format)
        if not predicate_element:
            # Not in the biolink model
            return None
        predicate_inverse_old_format = predicate_element.inverse
        if not predicate_inverse_old_format:
            # No inverse
            return None
        return self.old_case_to_new_case(predicate_inverse_old_format)

    @functools.cache
    def entity_prefixes(self, entity):
        """ Get prefixes for a given entity """
        old_format = self.new_case_to_old_case(entity)
        element = self.bmt.get_element(old_format)
        if not element:
            return None
        else:
            return element.id_prefixes




async def post_json(url, request):
    """Make post request."""
    async with httpx.AsyncClient(verify=False, timeout=None) as client:
        response = await client.post(
            url,
            json=request,
        )
        response.raise_for_status()
        return response.json()


class KPError(Exception):
    """Exception in a KP request."""


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


def listify_value(input_dictionary: dict[str, any], key: str):
    """ If the provided key is not a list, wrap it in a list """
    if key not in input_dictionary:
        return
    if isinstance(input_dictionary[key], str):
        input_dictionary[key] = [input_dictionary[key]]


def standardize_graph_lists(graph: dict[str, dict]):
    """ Convert fields that are given as a string to a list """
    node_fields = ['id', 'category']
    for node in graph['nodes'].values():
        for field in node_fields:
            listify_value(node, field)

    edge_fields = ['predicate']
    for edge in graph['edges'].values():
        for field in edge_fields:
            listify_value(edge, field)


def extract_predicate_direction(predicate: str) -> tuple[str, bool]:
    """ Extract predicate direction from string with enclosing arrows """
    if "<-" in predicate:
        return predicate[2:-1], True
    else:
        return predicate[1:-2], False


def build_predicate_direction(predicate: str, reverse: bool) -> str:
    """ Given a tuple of predicate string and direction, build a string with arrows """
    if reverse:
        return f"<-{predicate}-"
    else:
        return f"-{predicate}->"


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


def remove_null_values(obj):
    """Remove null values from all dicts in JSON-able object."""
    if isinstance(obj, dict):
        return {
            key: remove_null_values(value)
            for key, value in obj.items()
            if value is not None
        }
    elif isinstance(obj, list):
        return [
            remove_null_values(el)
            for el in obj
        ]
    else:
        return obj

def add_cors_manually(APP, request, response, cors_options):
    """
    Add CORS to a response manually
    Based on https://github.com/tiangolo/fastapi/issues/775
    """

    # Since the CORSMiddleware is not executed when an unhandled server exception
    # occurs, we need to manually set the CORS headers ourselves if we want the FE
    # to receive a proper JSON 500, opposed to a CORS error.
    # Setting CORS headers on server errors is a bit of a philosophical topic of
    # discussion in many frameworks, and it is currently not handled in FastAPI.
    # See dotnet core for a recent discussion, where ultimately it was
    # decided to return CORS headers on server failures:
    # https://github.com/dotnet/aspnetcore/issues/2378
    origin = request.headers.get('origin')

    if origin:
        # Have the middleware do the heavy lifting for us to parse
        # all the config, then update our response headers
        cors = CORSMiddleware(APP, **cors_options)

        # Logic directly from Starlette's CORSMiddleware:
        # https://github.com/encode/starlette/blob/master/starlette/middleware/cors.py#L152

        response.headers.update(cors.simple_headers)
        has_cookie = "cookie" in request.headers

        # If request includes any cookie headers, then we must respond
        # with the specific origin instead of '*'.
        if cors.allow_all_origins and has_cookie:
            response.headers["Access-Control-Allow-Origin"] = origin

        # If we only allow specific origins, then we have to mirror back
        # the Origin header in the response.
        elif not cors.allow_all_origins and cors.is_allowed_origin(origin=origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers.add_vary_header("Origin")

    return response

def get_from_all(
    dictionaries: list[dict],
    key,
    default=None
):
    """
    Get list of values from dictionaries.
    If it is not present in any dictionary, return the default value.
    """
    values = []
    for dictionary in dictionaries:
        value = dictionary.get(key, None)
        if value is not None:
            values.append(value)
    if len(values) > 0:
        return values
    else:
        return default

def merge_listify(values):
    """
    Merge values by converting them to lists
    and concatenating them.
    """
    output = []
    for value in values:
        if isinstance(value, list):
            output.extend(value)
        else:
            output.append(value)
    return output

def all_equal(values: list):
    """ Check that all values in given list are equal """
    return all(values[0] == v for v in values)

def deduplicate(values: list):
    """ Simple deduplicate that uses python sets """
    return list(set(values))
