""" Message store optimized for fast insertion of new messages """
import collections
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

def remove_null_attributes(d):
    """ Remove attributes property from dict if it is None """
    if d.get("attributes", []) is None:
        del d["attributes"]

def freeze_attribute(a):
    """ Freeze an attribute so that it can be hashed """
    a = frozendict(a)
    remove_null_attributes(a)
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
        # Freeze attributes
        for index in range(len(r["edge_bindings"][qg_id])):
            remove_null_attributes(
                r["edge_bindings"][qg_id][index]
            )
            if "attributes" in r["edge_bindings"][qg_id][index]:
                r["edge_bindings"][qg_id][index]["attributes"] = \
                    frozenset(
                        freeze_attribute(a)
                        for a in r["edge_bindings"][qg_id][index]["attributes"]
                    )

        r["edge_bindings"][qg_id] = frozenset(
            frozendict(eb)
            for eb in r["edge_bindings"][qg_id]
        )

    return r


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
        for curie, node in message["knowledge_graph"]["nodes"].items():
            key = frozendict({
                "id" : curie
            })

            # Access key so it is added to the list of keys
            self.nodes[key]

            # We don't really know how to merge names
            # so we just pick the first we are given
            if node.get("name", None):
                self.nodes[key]["name"] = node["name"]

            # Merge categories with a set
            if node.get("categories", None):
                self.nodes[key]["categories"].update(node["categories"])

            # Freeze attributes before adding so that
            # they can be deduplicated
            remove_null_attributes(node)
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

            # Access key so it is added to the list of keys
            self.edges[key]

            # Update mapping
            edge_id_mapping[frozendict({"id" : old_edge_id})] = key

            # Freeze attributes before adding so that
            # they can be deduplicated
            remove_null_attributes(edge)
            if "attributes" in edge:
                self.edges[key]["attributes"].update(
                    freeze_attribute(a) for a in edge["attributes"]
                )


        for result in message["results"]:
            def update_edge_binding(eb):
                """
                Replace a TRAPI kg_edge_id of format {"id" : X}
                with our custom format {"subject" : X, ...}
                """
                key = frozendict({ "id" : eb["id"] })
                # Lookup in dict
                new_eb = edge_id_mapping[key]
                # Copy over attributes
                if "attributes" in eb:
                    new_eb["attributes"] = eb["attributes"]
                return new_eb

            # Apply edge binding update function
            map_kg_edge_binding(result, update_edge_binding)

            # kg_node_ids happens to be in the same format as our custom keys
            # so we don't have to modify them

            # Freeze result
            result = freeze_result(result)

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
            output_message["knowledge_graph"]["edges"][get_hash_digest(edge_key)] = edge

        # Replace edge key in results with its hash
        for result in self.results:
            map_kg_edge_binding(result, lambda eb: {"id" : get_hash_digest(eb)})
            output_message["results"].append(result)

        return output_message
