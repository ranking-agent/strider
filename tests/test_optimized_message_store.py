import copy
import json
import pytest

from tests.helpers.utils import attribute_from_string
from strider.optimized_message_store import OptimizedMessageStore, freeze_attribute

class SetEncoder(json.JSONEncoder):
    """ JSON Encoder that works with sets """
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

def get_base_message():
    return {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {"nodes": {}, "edges": {}},
        "results": []
    }

@pytest.mark.asyncio
async def test_node_attribute_merging():
    """ Test that if nodes are merged node attributes are concatenated """

    ATTRIBUTE_A = attribute_from_string("""
        type biolink:knowledge_source value https://automat.renci.org/
            type biolink:has_p-value_evidence value 0.04
    """)

    ATTRIBUTE_B = attribute_from_string("""
        type biolink:publication value pubmed_central
            type biolink:has_original_source value true
    """)

    message_a = get_base_message()
    message_a["knowledge_graph"]["nodes"]["CHEBI:88916"] = {
        "attributes" : [ATTRIBUTE_A]
    }
    message_b = get_base_message()
    message_b["knowledge_graph"]["nodes"]["CHEBI:88916"] = {
        "attributes" : [ATTRIBUTE_B]
    }

    store = OptimizedMessageStore()

    store.add_message(message_a)
    store.add_message(message_b)
    output_message = store.get_message()

    # Use loads and dumps to convert sets -> lists
    output_attrs = json.loads(json.dumps(
        output_message["knowledge_graph"]["nodes"]["CHEBI:88916"]["attributes"],
        cls = SetEncoder,
    ))
    expected_attrs = json.loads(json.dumps(
        [ATTRIBUTE_B, ATTRIBUTE_A],
        cls = SetEncoder,
    ))
    assert output_attrs == expected_attrs

@pytest.mark.asyncio
async def test_result_merging():
    """ Test that duplicate results are merged correctly """

    message_a = get_base_message()
    message_a["knowledge_graph"]["edges"]["ke0"] = \
        {"subject" : "kn0", "object" : "kn1", "predicate" : "biolink:ameliorates"}
    message_a["results"].append({
        "node_bindings" : {"n0" : [{"id" : "kn0"}]},
        "edge_bindings" : {"e0" : [{"id" : "ke0"}]},
    })

    message_b = copy.deepcopy(message_a)

    store = OptimizedMessageStore()

    store.add_message(message_a)
    store.add_message(message_b)
    output_message = store.get_message()
    breakpoint()
