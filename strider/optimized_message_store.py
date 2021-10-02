""" Message store optimized for fast insertion of new messages """
import collections
import hashlib
import json

from reasoner_pydantic.message import Message
from reasoner_pydantic.base_model import FrozenDict

def get_hash_digest(o):
    """ Get the hash of an object as a human readable hex digest"""
    return hashlib.blake2b(
        hash(o).to_bytes(8, byteorder="big", signed = True),
        digest_size=6,
    ).hexdigest()

def map_kg_edge_binding(r, f):
    """ Apply a given mapping over each KG Edge binding """
    for qg_edge_id in r["edge_bindings"]:
        r["edge_bindings"][qg_edge_id] = [
            f(eb)
            for eb in r["edge_bindings"][qg_edge_id]
        ]

class OptimizedMessageStore():
    """ Message store optimized for fast insertion of new messages """

    def __init__(self):
        self.qgraph = {
            "nodes" : {},
            "edges" : {},
        }
        self.nodes = collections.defaultdict(
            lambda: {"categories" : set(), "attributes" : set()}
        )
        self.edges = collections.defaultdict(
            lambda: {"attributes" : set()}
        )
        self.results = set()

    def add_message(self, message: Message):
        # Freeze message
        message = Message.parse_obj(message).frozendict(setify = True)

        for curie, node in message["knowledge_graph"]["nodes"].items():
            key = FrozenDict(id = curie)

            # Access key so it is added to the list of keys
            self.nodes[key]

            # We don't really know how to merge names
            # so we just pick the first we are given
            if node.get("name", None):
                self.nodes[key]["name"] = node["name"]

            # Merge categories with a set
            if node.get("categories", None):
                self.nodes[key]["categories"].update(node["categories"])

            # Merge attributes with set
            if node.get("attributes", None):
                self.nodes[key]["attributes"].update(node["attributes"])

        # Mapping of old to new edge IDs
        edge_id_mapping = {}

        for old_edge_id, edge in message["knowledge_graph"]["edges"].items():
            key = FrozenDict(
                subject = edge["subject"],
                object  = edge["object"],
                predicate = edge["predicate"],
            )

            # Access key so it is added to the list of keys
            self.edges[key]

            # Update mapping
            edge_id_mapping[FrozenDict({"id" : old_edge_id})] = FrozenDict(key.copy())

            # Merge attributes with set
            if edge.get("attributes", None):
                self.edges[key]["attributes"].update(edge["attributes"])

        for result in message["results"]:
            def update_edge_binding(eb):
                """
                Replace a TRAPI kg_edge_id of format {"id" : X}
                with our custom format {"subject" : X, ...}
                """
                key = FrozenDict(id = eb["id"])
                # Lookup in dict
                new_eb = edge_id_mapping[key]
                # Copy over attributes
                if eb.get("attributes", None):
                    new_eb["attributes"] = eb["attributes"]
                return new_eb

            # Apply edge binding update function
            map_kg_edge_binding(result, update_edge_binding)

            # kg_node_ids happens to be in the same format as our custom keys
            # so we don't have to modify them

            # Add to set
            self.results.add(result)


    def get_message(self) -> Message:
        """ Convert message to output format"""
        output_message = {
            "query_graph" : self.qgraph,
            "knowledge_graph": {
                "nodes" : {},
                "edges" : {},
            },
            "results" : [],
        }

        for node_key, node in self.nodes.items():
            output_message["knowledge_graph"]["nodes"][node_key["id"]] = node

        for edge_key, edge in self.edges.items():
            # Add key properties to object
            edge.update(edge_key)
            # Replace edge key with its hash
            edge_key._key = None
            output_message["knowledge_graph"]["edges"][get_hash_digest(edge_key)] = edge

        def convert_to_hash_digest(eb):
            eb._key = None
            return FrozenDict(id = get_hash_digest(eb))

        # Replace edge key in results with its hash
        for result in self.results:
            map_kg_edge_binding(
                result,
                convert_to_hash_digest,
            )
            output_message["results"].append(result)

        return output_message
