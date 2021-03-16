from collections import defaultdict
import itertools
import time
import json
import re
import inspect
from strider.util import WrappedBMT
WBMT = WrappedBMT()


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
            "source_type": source,
            "edge_type": f"-{edge}->",
            "target_type": target,
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

    return {str(i): kp for i, kp in enumerate(kp_generator) if i < qty}


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
    edge_re = r"(?P<src>.*)-- (?P<predicate>.*) -->(?P<target>.*)"
    qg = {"nodes": {}, "edges": {}}
    for line in s.splitlines():
        match_node = re.search(node_re, line)
        match_edge = re.search(edge_re, line)
        if match_node:
            node_id = match_node.group('id')
            if node_id not in qg['nodes']:
                qg['nodes'][node_id] = {}
            qg['nodes'][node_id][match_node.group('key')] = \
                match_node.group('val')
        elif match_edge:
            edge_id = match_edge.group('src') + match_edge.group('target')
            qg['edges'][edge_id] = {
                "subject": match_edge.group('src'),
                "object": match_edge.group('target'),
                "predicate": match_edge.group('predicate'),
            }
        else:
            raise ValueError(f"Invalid line: {line}")
    return qg


def kps_from_string(s):
    """
    Converts a simple KP operation from a string format to JSON

    Example:
    kp0 biolink:ChemicalSubstance -biolink:treats-> biolink:Disease

    """
    s = inspect.cleandoc(s)
    kp_re = r"(?P<name>.*) (?P<src>.*) (?P<predicate>.*) (?P<target>.*)"
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
            "source_type": match_kp.group('src'),
            "edge_type": match_kp.group('predicate'),
            "target_type": match_kp.group('target'),
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
        action = tokens[1]
        line_data = tokens[2:]
        if action == 'categories':
            category_mappings[curie].extend(line_data)
        elif action == 'synonyms':
            synset_mappings[curie].extend(line_data)
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


async def time_and_display(f, msg):
    """ Time a function and print the time """
    start_time = time.time()
    await f()
    total = time.time() - start_time
    print("\n-------------------------------------------")
    print(f"Total time to {msg}: {total:.2f}s")
    print("-------------------------------------------")
