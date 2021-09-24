""" Message store optimized for fast insertion of new messages """
import collections
import copy

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


def freeze_attribute(a):
    """ Freeze an attribute so that it can be hashed """
    a = frozendict(a)
    if "attributes" in a:
        a["attributes"] = frozenset(
            freeze_attribute(sub_a) for sub_a in a["attributes"]
        )
    return a

class OptimizedMessageStore():
    """ Message store optimized for fast insertion of new messages """
    qgraph:  dict = {}
    nodes = collections.defaultdict(
        lambda: {"categories" : set(), "attributes" : set()}
    )
    edges:   dict = {}
    results: dict = {}
    node_bindings: dict = {}
    edge_bindings: dict = {}

    def add_message(self, message: Message):
        message = copy.deepcopy(message)
        for curie, node in message["knowledge_graph"]["nodes"].items():
            key = frozendict({ "id" : curie })

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

        for edge in message["knowledge_graph"].values():
            pass

    def get_message(self) -> Message:
        output_message = {
            "knowledge_graph": {
                "nodes" : {},
                "edges" : {},
            },
            "results" : [],
        }

        for node_key, node in self.nodes.items():
            output_message["knowledge_graph"]["nodes"][node_key["id"]] = node

        return output_message
