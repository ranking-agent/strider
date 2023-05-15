"""General utilities."""
import copy
import functools
import json
from json.decoder import JSONDecodeError
import re
from typing import Callable, Iterable, Union
import statistics

import httpx
from starlette.middleware.cors import CORSMiddleware
import yaml
import bmt
from bmt import Toolkit as BMToolkit
import logging.config
from strider.caching import save_post_request, get_post_response
from strider.config import settings


def camel_to_snake(s, sep=" "):
    return re.sub(r"(?<!^)(?=[A-Z])", sep, s).lower()


def snake_to_camel(s):
    return "".join(word.title() for word in s.split(" "))


class WrappedBMT:
    """
    Wrapping around some of the BMT Toolkit functions
    to provide case conversions to the new format
    """

    def __init__(self):
        self.bmt = BMToolkit()
        self.all_slots = self.bmt.get_all_slots()
        self.all_slots_formatted = [
            "biolink:" + s.replace(" ", "_") for s in self.all_slots
        ]
        self.prefix = "biolink:"

        self.entity_prefix_mapping = {
            bmt.util.format(el_name, case="pascal"): id_prefixes
            for el_name in self.bmt.get_all_classes()
            if (el := self.bmt.get_element(el_name)) is not None
            if (id_prefixes := getattr(el, "id_prefixes", []))
        }

    def new_case_to_old_case(self, s):
        """
        Convert new biolink case format (biolink:GeneOrGeneProduct)
        to old case format (gene or gene product)

        Also works with slots (biolink:related_to -> related to)
        """
        s = s.replace(self.prefix, "")
        if s in self.all_slots_formatted:
            return s.replace("_", " ")
        else:
            return camel_to_snake(s)

    def old_case_to_new_case(self, s):
        """
        Convert old case format (gene or gene product)
        to new biolink case format (biolink:GeneOrGeneProduct)

        Also works with slots (related to -> biolink:related_to)
        """
        if s in self.all_slots:
            return self.prefix + s.replace(" ", "_")
        else:
            return self.prefix + snake_to_camel(s)

    def get_descendants(self, concept):
        """Wrapped BMT descendants function that does case conversions"""
        descendants = self.bmt.get_descendants(concept, formatted=True)
        if len(descendants) == 0:
            descendants.append(concept)
        return descendants

    def get_ancestors(self, concept, reflexive=True):
        """Wrapped BMT ancestors function that does case conversions"""
        concept_old_format = self.new_case_to_old_case(concept)
        ancestors_old_format = self.bmt.get_ancestors(
            concept_old_format, reflexive=reflexive
        )
        ancestors = [self.old_case_to_new_case(a) for a in ancestors_old_format]
        return ancestors

    def predicate_is_symmetric(self, predicate):
        """Get whether a given predicate is symmetric"""
        predicate_old_format = self.new_case_to_old_case(predicate)
        predicate_element = self.bmt.get_element(predicate_old_format)
        if not predicate_element:
            # Not in the biolink model
            return False
        return predicate_element.symmetric

    def predicate_inverse(self, predicate):
        """Get the inverse of a predicate if it has one"""
        predicate_old_format = self.new_case_to_old_case(predicate)
        predicate_element = self.bmt.get_element(predicate_old_format)
        if not predicate_element:
            # Not in the biolink model
            return None
        if predicate_element.symmetric:
            return predicate
        predicate_inverse_old_format = predicate_element.inverse
        if not predicate_inverse_old_format:
            # No inverse
            return None
        return self.old_case_to_new_case(predicate_inverse_old_format)


WBMT = WrappedBMT()


def elide_curies(payload):
    """Elide CURIES in TRAPI request/response."""
    payload = copy.deepcopy(payload)
    if "nodes" in payload:  # qgraphs
        for qnode in payload["nodes"].values():
            if (num_curies := len(qnode.get("ids", None) or [])) > 10:
                qnode["ids"] = f"**{num_curies} CURIEs not shown for brevity**"
    if "curies" in payload:  # node norm responses
        payload["curies"] = f"**{len(payload['curies'])} CURIEs not shown for brevity**"
    if "message" in payload:  # messages
        for qnode in payload["message"]["query_graph"]["nodes"].values():
            if (num_curies := len(qnode.get("ids", None) or [])) > 10:
                qnode["ids"] = f"**{num_curies} CURIEs not shown for brevity**"
    return payload


def log_request(r):
    """Serialize a httpx.Request object into a dict for logging"""
    data = r.read().decode()
    # the request body can be cleared out by httpx under some circumstances
    # let's not crash if that happens
    try:
        data = elide_curies(json.loads(data))
    except Exception:
        pass
    return {
        "method": r.method,
        "url": str(r.url),
        "headers": dict(r.headers),
        "data": data,
    }


def log_response(r):
    """Serialize a httpx.Response object into a dict for logging"""
    return {
        "status_code": r.status_code,
        "headers": dict(r.headers),
        "data": r.text,
    }


class StriderRequestError(BaseException):
    """Custom error indicating an issue with an HTTP request"""


async def post_json(url, request, logger, log_name):
    """
    Make post request and write errors to log if present
    """
    # Commenting out cache because node norm is usually very quick and takes up
    # too much memory in the cache
    # response = await get_post_response(url, request)
    # if response is not None:
    #     # if we got back from cache
    #     return response
    # elif settings.offline_mode:
    #     logger.debug("POST JSON: Didn't get anything back from cache in offline mode.")
    #     # if not in cache and in offline mode
    #     return {}
    # else:
    try:
        async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
            logger.debug(f"Sending request to {url}")
            response = await client.post(
                url,
                json=request,
            )
            response.raise_for_status()
            response = response.json()
            # await save_post_request(url, request, response)
            return response
    except httpx.ReadTimeout as e:
        logger.warning(
            {
                "message": f"{log_name} took >60 seconds to respond",
                "error": str(e),
                "request": log_request(e.request),
            }
        )
    except httpx.RequestError as e:
        # Log error
        logger.warning(
            {
                "message": f"Request Error contacting {log_name}",
                "error": str(e),
                "request": log_request(e.request),
            }
        )
    except httpx.HTTPStatusError as e:
        # Log error with response
        logger.warning(
            {
                "message": f"Response Error contacting {log_name}",
                "error": str(e),
                "request": log_request(e.request),
                "response": log_response(e.response),
            }
        )
    except JSONDecodeError as e:
        # Log error with response
        logger.warning(
            {
                "message": f"Received bad JSON data from {log_name}",
                "request": e.request,
                "response": e.response.text,
                "error": str(e),
            }
        )
    except Exception as e:
        # General catch all
        logger.warning(
            {
                "message": f"Something went wrong when contacting {log_name}.",
                "error": str(e),
            }
        )
    raise StriderRequestError


class KnowledgeProvider:
    """Knowledge provider."""

    def __init__(self, details, portal, id, *args, **kwargs):
        """Initialize."""
        self.details = details
        self.portal = portal
        # self.portal: KnowledgePortal = portal
        self.id = id

    async def solve_onehop(self, request):
        """Solve one-hop query."""
        return await self.portal.fetch(
            self.id,
            {"message": {"query_graph": request}},
        )


class KPError(Exception):
    """Exception in a KP request."""


def setup_logging():
    """Set up logging."""
    with open("logging_setup.yml", "r") as stream:
        config = yaml.load(stream.read(), Loader=yaml.SafeLoader)
    logging.config.dictConfig(config)


def ensure_list(arg: Union[str, list[str]]) -> list[str]:
    """Enclose in list if necessary."""
    if isinstance(arg, list):
        return arg
    return [arg]


def batch(iterable: Iterable, n: int = 1):
    """Batch things into batches of size n."""
    N = len(iterable)
    for ndx in range(0, N, n):
        yield iterable[ndx : min(ndx + n, N)]


def listify_value(input_dictionary: dict[str, any], key: str):
    """If the provided key is not a list, wrap it in a list"""
    if key not in input_dictionary:
        return
    if isinstance(input_dictionary[key], str):
        input_dictionary[key] = [input_dictionary[key]]


def extract_predicate_direction(predicate: str) -> tuple[str, bool]:
    """Extract predicate direction from string with enclosing arrows"""
    if "<-" in predicate:
        return predicate[2:-1], True
    else:
        return predicate[1:-2], False


def build_predicate_direction(predicate: str, reverse: bool) -> str:
    """Given a tuple of predicate string and direction, build a string with arrows"""
    if reverse:
        return f"<-{predicate}-"
    else:
        return f"-{predicate}->"


def message_to_list_form(message):
    """Convert *graph nodes/edges and node/edge bindings to list forms."""
    if message["results"]:
        message["results"] = [
            {
                "node_bindings": [
                    {
                        "qg_id": qg_id,
                        **binding,
                    }
                    for qg_id, binding in result["node_bindings"].items()
                ],
                "analyses": [
                    {
                        "resource_id": "infores:aragorn",
                        "edge_bindings": [
                            {
                                "qg_id": qg_id,
                                **binding,
                            }
                            for qg_id, binding in analysis["edge_bindings"].items()
                        ],
                    }
                    for analysis in result.get("analyses",[])
                ]
            }
            for result in message.get("results", [])
        ]
    if message["knowledge_graph"]["nodes"]:
        message["knowledge_graph"]["nodes"] = [
            {
                "id": node["id"],
                **node,
            }
            for node in message["knowledge_graph"]["nodes"]
        ]
    if message["knowledge_graph"]["edges"]:
        message["knowledge_graph"]["edges"] = [
            {
                "id": edge["id"],
                **edge,
            }
            for edge in message["knowledge_graph"]["edges"]
        ]
    return message


def message_to_dict_form(message):
    """Convert *graph nodes/edges and node/edge bindings to dict forms."""
    if message["results"]:
        if isinstance(message["results"][0]["node_bindings"], list):
            message["results"] = [
                {
                    "node_bindings": {
                        binding["qg_id"]: [binding]
                        for binding in result["node_bindings"]
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:aragorn",
                            "edge_bindings": {
                                binding["qg_id"]: [binding]
                                for binding in analysis["edge_bindings"]
                            },
                        }
                        for analysis in result.get("analyses", [])
                    ]
                }
                for result in message.get("results", [])
            ]
        elif not isinstance(
            list(message["results"][0]["node_bindings"].values())[0], dict
        ):
            message["results"] = [
                {
                    "node_bindings": {
                        key: [{"kg_id": el} for el in ensure_list(bindings)]
                        for key, bindings in result["node_bindings"].items()
                    },
                    "analyses": [
                        {
                            "resource_id": "infores:aragorn",
                            "edge_bindings": {
                                key: [{"kg_id": el} for el in ensure_list(bindings)]
                                for key, bindings in analysis["edge_bindings"].items()
                            },
                        }
                        for analysis in result.get("analyses", [])
                    ]
                }
                for result in message.get("results", [])
            ]
    if message["knowledge_graph"]["nodes"]:
        message["knowledge_graph"]["nodes"] = {
            node["id"]: node for node in message["knowledge_graph"]["nodes"]
        }
    if message["knowledge_graph"]["edges"]:
        message["knowledge_graph"]["edges"] = {
            edge["id"]: edge for edge in message["knowledge_graph"]["edges"]
        }
    return message


def graph_to_dict_form(graph):
    """Convert query_graph or knowledge_graph to dict form."""
    return {
        "nodes": {node["id"]: node for node in graph["nodes"]},
        "edges": {edge["id"]: edge for edge in graph["edges"]},
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
        return [remove_null_values(el) for el in obj]
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
    origin = request.headers.get("origin")

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


def get_from_all(dictionaries: list[dict], key, default=None):
    """
    Get list of values from dictionaries.
    If it is not present in any dictionary, return the default value.
    """
    values = [d[key] for d in dictionaries if key in d]
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


def filter_none(values):
    """Filter out None values from list"""
    return [v for v in values if v is not None]


def all_equal(values: list):
    """Check that all values in given list are equal"""
    return all(values[0] == v for v in values)


def deduplicate(values: list):
    """Simple deduplicate that uses python sets"""
    return list(set(values))


def transform_keys(d, f):
    """Transform keys using a function"""
    return {f(key): val for key, val in d.items()}


def get_message_stats(m):
    """Get statistics on message size"""
    stats = {}

    stats["nodes"] = len(m["knowledge_graph"]["nodes"])
    stats["edges"] = len(m["knowledge_graph"]["edges"])

    stats["avg_node_categories"] = statistics.mean(
        len(n["categories"]) for n in m["knowledge_graph"]["nodes"].values()
    )

    stats["avg_kg_attributes"] = statistics.mean(
        len(n["attributes"]) if n["attributes"] else 0
        for n in m["knowledge_graph"]["nodes"].values()
    )
    stats["avg_kg_attributes"] += statistics.mean(
        len(e["attributes"]) if e["attributes"] else 0
        for e in m["knowledge_graph"]["edges"].values()
    )

    stats["results"] = len(m["results"])
    stats["avg_result_node_bindings"] = statistics.mean(
        len(nb_list) for r in m["results"] for nb_list in r["node_bindings"].values()
    )
    stats["avg_result_edge_bindings"] = statistics.mean(
        len(nb_list) for r in m["results"] for nb_list in r["node_bindings"].values()
    )
    return stats


def get_kp_operations_queries(
    subject_categories: Union[str, list[str]] = None,
    predicates: Union[str, list[str]] = None,
    object_categories: Union[str, list[str]] = None,
):
    """Build queries to send to kp registry."""
    if isinstance(subject_categories, str):
        subject_categories = [subject_categories]
    if isinstance(predicates, str):
        predicates = [predicates]
    if isinstance(object_categories, str):
        object_categories = [object_categories]
    subject_categories = [
        desc for cat in subject_categories for desc in WBMT.get_descendants(cat)
    ]
    predicates = [desc for pred in predicates for desc in WBMT.get_descendants(pred)]
    inverse_predicates = [
        desc
        for pred in predicates
        if (inverse := WBMT.predicate_inverse(pred))
        for desc in WBMT.get_descendants(inverse)
    ]
    object_categories = [
        desc for cat in object_categories for desc in WBMT.get_descendants(cat)
    ]

    return subject_categories, object_categories, predicates, inverse_predicates
