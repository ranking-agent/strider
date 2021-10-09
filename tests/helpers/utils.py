from collections import defaultdict
import inspect
import itertools
import time
import json
import random
import re
import string

from reasoner_pydantic.message import Message
from reasoner_pydantic.results import NodeBinding, EdgeBinding, Result
from reasoner_pydantic.qgraph import QueryGraph
from reasoner_pydantic.kgraph import Edge, KnowledgeGraph, Node

from strider.util import WBMT


def load_kps(fpath):
    """ Load KPs from a file for use in a test """
    with open(fpath, "r") as f:
        kps = json.load(f)
    DEFAULT_PREFIXES = {
        "biolink:Disease": ["MONDO", "DOID"],
        "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
        "biolink:PhenotypicFeature": ["HP"],
    }
    # Add prefixes
    for kp in kps.values():
        kp['details'] = {'preferred_prefixes': DEFAULT_PREFIXES}
    return kps


def create_kp(args):
    """
    Generate a KP given a tuple of input, edge, output
    """
    source, edge, target = args
    return {
        "url": "http://mykp",
        "operations": [{
            "subject_category": source,
            "predicate": f"-{edge}->",
            "object_category": target,
        }]
    }


def generate_kps(qty):
    """
    Generate a given number of KPs using permutations
    of the biolink model
    """
    node_categories = WBMT.get_descendants('biolink:NamedThing')
    edge_predicates = WBMT.get_descendants('biolink:related_to')
    kp_generator = map(
        create_kp,
        itertools.product(
            node_categories,
            edge_predicates,
            node_categories,
        )
    )

    return {str(i): kp for i, kp in zip(range(qty), kp_generator)}


def generate_message(spec) -> Message:
    """
    Generate a message using a specification. Example for the specification format:

    {
        "knowledge_graph" : {
            "nodes" : {
                "count" : 100,
                "attributes" : {
                    "count" : 100,
                    "value_type" : fn
                },
                "categories_count" : 1
            },
            "edges" : {
                "count" : 100,
                "attributes" : {
                    "count" : 100,
                    "generator" : fn
                }
            }
        },
        "results" : {
            "count" : 100,
            "node_bindings" : {
                "count_per_node" : 1
            },
            "edge_bindings" : {
                "count_per_edge" : 1,
                "attributes" : {
                    "count" : 100,
                    "generator" : fn
                }
            }
        }
    }
    """

    get_random = lambda: "".join(random.choice(string.ascii_letters) for _ in range(10))

    kg_node_ids = [
       f"biolink:{get_random()}"
       for _ in range(spec["knowledge_graph"]["nodes"]["count"])
    ]
    kg_edge_ids = [
       get_random()
       for _ in range(spec["knowledge_graph"]["nodes"]["count"])
    ]

    return Message(
        query_graph = QueryGraph(nodes = {}, edges = {}),
        knowledge_graph = KnowledgeGraph(
            nodes = {
                kgnid : Node(
                    attributes = list(itertools.islice(
                        spec["knowledge_graph"]["nodes"]["attributes"]["generator"],
                        spec["knowledge_graph"]["nodes"]["attributes"]["count"],
                    )),
                    categories = [
                        f"biolink:Category{get_random()}"
                        for _ in range(spec["knowledge_graph"]["nodes"]["categories_count"])
                    ],
                )
                for kgnid in kg_node_ids
            },
            edges = {
                kgeid : Edge(
                    attributes = list(itertools.islice(
                        spec["knowledge_graph"]["edges"]["attributes"]["generator"],
                        spec["knowledge_graph"]["edges"]["attributes"]["count"],
                    )),
                    subject = random.choice(kg_node_ids),
                    predicate = f"biolink:{get_random().lower()}",
                    object = random.choice(kg_node_ids),
                )
                for kgeid in kg_edge_ids
            },
        ),
        results = [
            Result(
                node_bindings = {
                    f"QGraphNode:{get_random()}" : [
                        NodeBinding(
                            id = kgnid
                        )
                        for _ in range(spec["results"]["node_bindings"]["count_per_node"])
                    ]
                    for kgnid in kg_node_ids
                },
                edge_bindings = {
                    f"QGraphEdge:{get_random()}" : [
                        EdgeBinding(
                            id = kgeid,
                            attributes = list(itertools.islice(
                                spec["results"]["edge_bindings"]["attributes"]["generator"],
                                spec["results"]["edge_bindings"]["attributes"]["count"],
                            )),
                        )
                        for _ in range(spec["results"]["edge_bindings"]["count_per_edge"])
                    ]
                    for kgeid in kg_edge_ids
                },
            )
            for _ in range(spec["results"]["count"])
        ],
    )


def query_graph_from_string(s):
    """
    Parse a query graph from Mermaid flowchart syntax.
    Useful for writing examples in tests.

    Syntax information can be found here:
    https://mermaid-js.github.io/mermaid/#/flowchart

    You can use this site to preview a query graph:
    https://mermaid-js.github.io/mermaid-live-editor/
    """

    # This usually comes from triple quoted strings
    # so we use inspect.cleandoc to remove leading indentation
    s = inspect.cleandoc(s)

    node_re = r"(?P<id>.*)\(\( (?P<key>.*) (?P<val>.*) \)\)"
    edge_re = r"(?P<src>.*)-- (?P<predicates>.*) -->(?P<target>.*)"
    qg = {"nodes": {}, "edges": {}}
    for line in s.splitlines():
        match_node = re.search(node_re, line)
        match_edge = re.search(edge_re, line)
        if match_node:
            node_id = match_node.group('id')
            node = qg["nodes"].get(node_id, dict())
            if node_id not in qg['nodes']:
                qg['nodes'][node_id] = node
            key = match_node.group('key')
            if key.endswith("[]"):
                node[key[:-2]] = \
                    node.get(key[:-2], []) + [match_node.group('val')]
            else:
                node[key] = \
                    match_node.group('val')
        elif match_edge:
            edge_id = match_edge.group('src') + match_edge.group('target')
            qg['edges'][edge_id] = {
                "subject": match_edge.group('src'),
                "object": match_edge.group('target'),
                "predicates": match_edge.group('predicates').split(" "),
            }
        else:
            raise ValueError(f"Invalid line: {line}")
    return qg


def kps_from_string(s):
    """
    Converts a simple KP operation from a string format to JSON

    Example:
    kp0 biolink:ChemicalSubstance biolink:treats biolink:Disease

    """
    s = inspect.cleandoc(s)
    kp_re = r"(?P<name>.*) (?P<subject>.*) (?P<predicate>.*) (?P<object>.*)"
    kps = {}
    for line in s.splitlines():
        match_kp = re.search(kp_re, line)
        if not match_kp:
            raise ValueError(f"Invalid line: {line}")
        name = match_kp.group('name')
        if name not in kps:
            kps[name] = {
                "url": f"http://{name}",
                "operations": [],
            }
        kps[name]['operations'].append({
            "subject_category": match_kp.group('subject'),
            "predicate": match_kp.group('predicate'),
            "object_category": match_kp.group('object'),
        })
    return kps


def normalizer_data_from_string(s):
    """
    Parse data for node normalizer from string. Useful for tests.

    Basic syntax:

    MONDO:0005737 categories biolink:Disease
    MONDO:0005737 synonyms   DOID:4325 ORPHANET:319218
    """

    # This usually comes from triple quoted strings
    # so we use inspect.cleandoc to remove leading indentation
    s = inspect.cleandoc(s)

    category_mappings = defaultdict(list)
    synset_mappings = defaultdict(list)
    for line in s.splitlines():
        tokens = line.split(" ")
        curie = tokens[0]

        # Curies are always self-synonyms
        if curie not in synset_mappings[curie]:
            synset_mappings[curie].append(curie)

        action = tokens[1]
        line_data = tokens[2:]

        if action == 'categories':
            category_mappings[curie].extend(line_data)
        elif action == 'synonyms':
            # Add to start of list so that we can override
            # the primary CURIE
            synset_mappings[curie] = sorted(line_data + synset_mappings[curie])
        else:
            raise ValueError(f"Invalid line: {line}")

    return {"category_mappings": category_mappings, "synset_mappings": synset_mappings}


def plan_template_from_string(s):
    """
    Create a template to validate a plan from a string format

    Example:

    n0-n0n1-n1 http://kp0 biolink:Drug -biolink:related_to-> biolink:Disease
    n0-n0n1-n1 http://kp1 biolink:Drug -biolink:treats-> biolink:Disease
    n1-n1n2-n2 http://kp2 biolink:Disease -biolink:has_phenotype-> biolink:PhenotypicFeature

    """
    # This usually comes from triple quoted strings
    # so we use inspect.cleandoc to remove leading indentation
    s = inspect.cleandoc(s)

    plan_template = defaultdict(list)

    for line in s.splitlines():
        tokens = line.split(" ")
        step = tuple(tokens[0].split("-"))

        plan_template[step].append({
            "url": tokens[1],
            "source_category": tokens[2],
            "edge_predicate": tokens[3],
            "target_category": tokens[4],
        })

    return dict(plan_template)


def validate_template(template, value):
    """
    Assert that value adheres to the provided template

    This means that:
    1. All dictionary key-value pairs in the template are
       present and equal to instance values
    2. All lists must be of the same length and have equal
       values to the template
    3. All values that are not lists or dictionaries
       must compare equal using Python equality

    """
    if isinstance(template, list):
        if len(template) != len(value):
            raise ValueError("Lists are not the same length")
        for index in range(len(template)):
            validate_template(template[index], value[index])
    elif isinstance(template, dict):
        for key in template.keys():
            if key not in value:
                raise ValueError(f"Key {key} not present in value")
            validate_template(template[key], value[key])
    else:
        if template != value:
            raise ValueError(
                f"Template value {template} does not equal {value}")


def validate_message(template, value):
    """
    Validate a message against the given template

    Raises ValueError if anything doesn't match up.

    Templates must be a dictionary with "knowledge_graph" and "results"
    keys. Knowledge_graph should be a string representation of the knowledge
    graph edges. Results should be a list of strings each representing a results object.
    """

    template["knowledge_graph"] = inspect.cleandoc(template["knowledge_graph"])

    nodes = set()
    # Validate edges
    for edge_string in template["knowledge_graph"].splitlines():
        sub, predicate, obj = edge_string.split(" ")
        nodes.add(sub)
        nodes.add(obj)
        # Check that this edge exists
        if not any(
                edge["subject"] == sub and
                edge["object"] == obj and
                predicate in edge["predicate"]
            for edge in value["knowledge_graph"]["edges"].values()
        ):
            raise ValueError(
                f"Knowledge graph edge {edge_string} not found in message")

    # Validate nodes
    for node in nodes:
        if node not in value["knowledge_graph"]["nodes"].keys():
            raise ValueError(
                f"Knowledge graph node {node} not found in message")

    # Check for extra nodes or edges
    if len(nodes) != len(value["knowledge_graph"]["nodes"]):
        raise ValueError(
            "Extra nodes found in message knowledge_graph")
    if (
        len(template["knowledge_graph"].splitlines()) !=
        len(value["knowledge_graph"]["edges"])
    ):
        raise ValueError(
            "Extra edges found in message knowledge_graph")

    # Validate results
    for index, template_result_string in enumerate(template["results"]):

        # Parse string representation
        template_result_string = inspect.cleandoc(template_result_string)
        template_result = {}
        current_key = None
        for line in template_result_string.splitlines():
            if not line.startswith(" "):
                # Key (remove trailing colon)
                current_key = line[:-1]
                template_result[current_key] = []
            else:
                # Value
                template_result[current_key].append(line.strip())

        for value_result in value["results"]:
            try:
                # Validate node bindings
                for node_binding_string in template_result["node_bindings"]:
                    qg_node_id, *kg_node_ids = node_binding_string.split(" ")
                    if qg_node_id not in value_result["node_bindings"]:
                        raise ValueError(
                            f"Could not find binding for node {qg_node_id}")

                    for kg_node_id in kg_node_ids:
                        if not any(
                            nb["id"] == kg_node_id
                            for nb in value_result["node_bindings"][qg_node_id]
                        ):
                            raise ValueError(
                                f"Expected node binding {qg_node_id} to {kg_node_id}")
                    if len(value_result["node_bindings"][qg_node_id]) != len(kg_node_ids):
                        raise ValueError(f"Extra node bindings found for {qg_node_id}")

                # Validate edge bindings
                for edge_binding_string in template_result["edge_bindings"]:
                    qg_edge_id, *kg_edge_strings = edge_binding_string.split(" ")

                    # Find KG edge IDs from the kg edge strings
                    kg_edge_ids = []
                    for kg_edge_string in kg_edge_strings:
                        sub, obj = kg_edge_string.split("-")
                        kg_edge_id = next(
                            kg_edge_id
                            for kg_edge_id, kg_edge in value["knowledge_graph"]["edges"].items()
                            if kg_edge["subject"] == sub and kg_edge["object"] == obj
                        )
                        kg_edge_ids.append(kg_edge_id)

                    if qg_edge_id not in value_result["edge_bindings"]:
                        raise ValueError(
                            f"Could not find binding for edge {qg_edge_id}")

                    for kg_edge_id in kg_edge_ids:
                        if not any(
                            nb["id"] == kg_edge_id
                            for nb in value_result["edge_bindings"][qg_edge_id]
                        ):
                            raise ValueError(
                                f"Expected edge binding {qg_edge_id} to {kg_edge_id}")
                    if len(value_result["edge_bindings"][qg_edge_id]) != len(kg_edge_ids):
                        raise ValueError(f"Extra edge bindings found for {qg_edge_id}")
            except ValueError as err:
                continue
            break
        else:
            raise err

    # Check for extra results
    if len(template["results"]) != len(value["results"]):
        raise ValueError("Extra results found")


async def time_and_display(f, msg):
    """ Time a function and print the time """
    start_time = time.time()
    await f()
    total = time.time() - start_time
    print("\n-------------------------------------------")
    print(f"Total time to {msg}: {total:.2f}s")
    print("-------------------------------------------")
