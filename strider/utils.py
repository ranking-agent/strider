"""General utilities."""

import copy
import json
from json.decoder import JSONDecodeError
import re
from reasoner_pydantic import Message
from typing import Iterable

import httpx
from starlette.middleware.cors import CORSMiddleware
import yaml
from bmt.toolkit import Toolkit as BMToolkit
import logging.config


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
            el_name: id_prefixes
            for el_name in self.bmt.get_all_classes(formatted=True)
            if (el := self.bmt.get_element(el_name)) is not None
            if (id_prefixes := getattr(el, "id_prefixes", []))
        }

    def get_descendants(self, concept, formatted=True, reflexive=True):
        """Wrapped BMT descendants function that does case conversions"""
        if self.bmt.get_element(concept) is not None:
            return self.bmt.get_descendants(
                concept, formatted=formatted, reflexive=reflexive
            )
        return [concept]

    def get_ancestors(self, concept, formatted=True, reflexive=True):
        """Wrapped BMT ancestors function that does case conversions"""
        if self.bmt.get_element(concept) is not None:
            return self.bmt.get_ancestors(
                concept, formatted=formatted, reflexive=reflexive
            )
        return [concept]

    def predicate_is_symmetric(self, predicate):
        """Get whether a given predicate is symmetric"""
        return self.bmt.is_symmetric(predicate)

    def predicate_inverse(self, predicate, formatted=True):
        """Get the inverse of a predicate if it has one"""
        return self.bmt.get_inverse_predicate(predicate, formatted=formatted)


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
    if "query_graph" in payload:  # TRAPI message
        for qnode in payload["query_graph"]["nodes"].values():
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
    try:
        async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
            logger.debug(f"Sending request to {url}")
            response = await client.post(
                url,
                json=request,
            )
            response.raise_for_status()
            response = response.json()
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


def setup_logging():
    """Set up logging."""
    with open("logging_setup.yml", "r") as stream:
        config = yaml.load(stream.read(), Loader=yaml.SafeLoader)
    logging.config.dictConfig(config)


def batch(iterable: Iterable, n: int = 1):
    """Batch things into batches of size n."""
    N = len(iterable)
    for ndx in range(0, N, n):
        yield iterable[ndx : min(ndx + n, N)]


def get_curies(message: Message) -> list[str]:
    """Get all node curies used in message.

    Do not examine kedge source and target ids. There ought to be corresponding
    knodes.
    """
    curies = set()
    if message.query_graph is not None:
        for qnode in message.query_graph.nodes.values():
            if qnode.ids:
                curies |= set(qnode.ids)
    if message.knowledge_graph is not None:
        curies |= set(message.knowledge_graph.nodes.keys())
    return list(curies)


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
