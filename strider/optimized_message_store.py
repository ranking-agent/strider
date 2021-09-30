""" Message store optimized for fast insertion of new messages """
import collections
import hashlib
import json

from reasoner_pydantic.message import Message

class frozendict(dict):
    """
    Dict class that can be used as a key (hashable)

    This class provides NO enforcement for mutation
    """
    def __init__(self, *args, **kwargs):
        self._key = None
        super().__init__(*args, **kwargs)
    def __key(self):
        # Use cache for key
        if not self._key:
            self._key =  tuple((k,self[k]) for k in sorted(self))
        return self._key
    def __hash__(self):
        return hash(self.__key())
    def __eq__(self, other):
        return self.__key() == other.__key()

class CustomizableJSONDecoder(json.JSONDecoder):
    """ JSON Decoder that allows a custom list_type """
    def __init__(self, list_type=list,  **kwargs):
        json.JSONDecoder.__init__(self, **kwargs)
        # Use the custom JSONArray
        self.parse_array = self.JSONArray
        # Use the python implemenation of the scanner
        self.scan_once = json.scanner.py_make_scanner(self)
        self.list_type=list_type

    def JSONArray(self, s_and_end, scan_once, **kwargs):
        values, end = json.decoder.JSONArray(s_and_end, scan_once, **kwargs)
        return self.list_type(values), end

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

def freeze_object(o):
    """ Freeze object using json dump and loads """

    def set_default(obj):
        """
        Dump json with support for frozenset
        This only happens during tests
        """
        if isinstance(obj, frozenset) or isinstance(obj, set):
            return list(obj)
        raise TypeError
    o_json = json.dumps(o, default = set_default)

    def remove_null_frozendict(dct):
        """ Remove None values and convert to frozendict"""
        not_null_iterator = {k:v for k,v in dct.items() if v != None}
        return frozendict(not_null_iterator)

    return json.loads(
        o_json,
        cls = CustomizableJSONDecoder,
        list_type = frozenset,
        object_hook = remove_null_frozendict,
    )


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
        message = freeze_object(message)

        for curie, node in message["knowledge_graph"]["nodes"].items():
            key = frozendict(id = curie)

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
            key = frozendict(
                subject = edge["subject"],
                object  = edge["object"],
                predicate = edge["predicate"],
            )

            # Access key so it is added to the list of keys
            self.edges[key]

            # Update mapping
            edge_id_mapping[frozendict({"id" : old_edge_id})] = frozendict(key.copy())

            # Merge attributes with set
            if edge.get("attributes", None):
                self.edges[key]["attributes"].update(edge["attributes"])

        for result in message["results"]:
            def update_edge_binding(eb):
                """
                Replace a TRAPI kg_edge_id of format {"id" : X}
                with our custom format {"subject" : X, ...}
                """
                key = frozendict(id = eb["id"])
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
            return frozendict(id = get_hash_digest(eb))

        # Replace edge key in results with its hash
        for result in self.results:
            map_kg_edge_binding(
                result,
                convert_to_hash_digest,
            )
            output_message["results"].append(result)

        return output_message
