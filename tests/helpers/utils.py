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
            qg['nodes'][match_node.group('id')] = {
                match_node.group('key'): match_node.group('val')
            }
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


async def time_and_display(f, msg):
    """ Time a function and print the time """
    start_time = time.time()
    await f()
    total = time.time() - start_time
    print("\n-------------------------------------------")
    print(f"Total time to {msg}: {total:.2f}s")
    print("-------------------------------------------")
