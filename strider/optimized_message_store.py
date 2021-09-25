""" Message store optimized for fast insertion of new messages """
import collections
import copy
import hashlib

from reasoner_pydantic.message import Message

class frozendict(dict):
    """
    Dict class that can be used as a key (hashable)

    This class provides NO enforcement for mutation
    """
    def __key(self):
        return tuple((k,self[k]) for k in sorted(self))
    def __hash__(self):
        return hash(self.__key())
    def __eq__(self, other):
        return self.__key() == other.__key()


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

def freeze_attribute(a):
    """ Freeze an attribute so that it can be hashed """
    a = frozendict(a)
    if "attributes" in a:
        a["attributes"] = frozenset(
            freeze_attribute(sub_a) for sub_a in a["attributes"]
        )
    return a

def freeze_result(r):
    """ Freeze a result so it can be hashed"""
    r = frozendict(r)
    r["node_bindings"] = frozendict(r["node_bindings"])
    r["edge_bindings"] = frozendict(r["edge_bindings"])

    for qg_id in r["node_bindings"].keys():
        r["node_bindings"][qg_id] = frozenset(
            frozendict(nb)
            for nb in r["node_bindings"][qg_id]
        )
    for qg_id in r["edge_bindings"].keys():
        r["edge_bindings"][qg_id] = frozenset(
            frozendict(eb)
            for eb in r["edge_bindings"][qg_id]
        )
    return r


class OptimizedMessageStore():
    """ Message store optimized for fast insertion of new messages """
    qgraph:  dict = {}
    nodes = collections.defaultdict(
        lambda: {"categories" : set(), "attributes" : set()}
    )
    edges = collections.defaultdict(
        lambda: {"attributes" : set()}
    )
    results = set()

    def add_message(self, message: Message):
        message = copy.deepcopy(message)
        for curie, node in message["knowledge_graph"]["nodes"].items():
            key = frozendict({
                "id" : curie
            })

            # We don't really know how to merge names
            # so we just pick the first we are given
            if "name" in node:
                self.nodes[key]["name"] = node["name"]

            if "categories" in node:
                self.nodes[key]["categories"].update(node["categories"])

            # Freeze attributes before adding so that
            # they can be deduplicated
            if "attributes" in node:
                self.nodes[key]["attributes"].update(
                    freeze_attribute(a) for a in node["attributes"]
                )


        # Mapping of old to new edge IDs
        edge_id_mapping = {}

        for old_edge_id, edge in message["knowledge_graph"]["edges"].items():
            key = frozendict({
                "subject" : edge["subject"],
                "object"  : edge["object"],
                "predicate" : edge["predicate"],
            })

            edge_id_mapping[frozendict({"id" : old_edge_id})] = key

            # Freeze attributes before adding so that
            # they can be deduplicated
            if "attributes" in edge:
                self.edges[key]["attributes"].update(
                    freeze_attribute(a) for a in edge["attributes"]
                )


        # Replace kg_edge_id with our custom key format
        for result in message["results"]:
            map_kg_edge_binding(result, lambda eb: edge_id_mapping[frozendict(eb)])

        # kg_node_ids happen to be in the same format as our custom keys
        # so we don't have to modify them

        # Add to set of results
        self.results.update(
            freeze_result(r) for r in message["results"]
        )

    def get_message(self) -> Message:
        """ Convert message to output format"""
        output_message = {
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
            output_message["knowledge_graph"]["edges"][get_hash_digest(edge_key)] = edge

        # Replace edge key in results with its hash
        for result in self.results:
            result = copy.deepcopy(result)
            map_kg_edge_binding(result, get_hash_digest)
            output_message["results"].append(result)

        return output_message
