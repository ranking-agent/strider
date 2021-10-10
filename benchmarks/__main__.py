import itertools
import random
import time

import seaborn
from reasoner_pydantic.shared import Attribute

from benchmarks.utils import benchmark_parameterized, find_polynomial_coefficients
from strider.trapi import merge_messages
from tests.helpers.utils import generate_message

def randint_generator(a, b):
    while True:
        yield random.randint(a, b)

def merge_message_parameterized(
        kg_node_count,
        kg_node_categories_count,
        kg_edge_count,
        kg_attribute_count,

        result_count,
        result_attribute_count,

        attribute_value_size,
        attribute_subattribute_count,
):
    """
    Merge_messages wrapper for benchmarking
    """

    attribute_spec = {
        "subattribute_count" : attribute_subattribute_count,
        "value_type" : "list",
        "value_count" : attribute_value_size,
    }

    messages = [generate_message({
        "knowledge_graph" : {
            "nodes" : {
                "count" : kg_node_count,
                "attributes" : {
                    "count" : kg_attribute_count,
                    "spec" : attribute_spec
                },
                "categories_count" : kg_node_categories_count
            },
            "edges" : {
                "count" : kg_edge_count,
                "attributes" : {
                    "count" : kg_attribute_count,
                    "spec" : attribute_spec
                }
            }
        },
        "results" : {
            "count" : result_count,
            "node_bindings" : {
                "count_per_node" : 1
            },
            "edge_bindings" : {
                "count_per_edge" : 1,
                "attributes" : {
                    "count" : result_attribute_count,
                    "spec" : attribute_spec
                }
            }
        }
    }).dict() for _ in range(2)]

    merge_messages(messages)

params = {
    "kg_node_count" : randint_generator(1, 128),
    "kg_node_categories_count" : randint_generator(1, 128),
    "kg_edge_count" : randint_generator(1, 128),
    "kg_attribute_count" : randint_generator(1, 16),

    "result_count" : randint_generator(1, 128),
    "result_attribute_count" : randint_generator(1, 16),

    "attribute_value_size" : randint_generator(1, 16),
    "attribute_subattribute_count" : randint_generator(1, 16),
}

# Run benchmark
results = benchmark_parameterized(
    fn = merge_message_parameterized,
    parameters = params,
    iterations = 1000,
    repititions = 1,
)

# Analyze data
poly_coef = find_polynomial_coefficients(results)

# Create heat map and output
seaborn.set_theme()
plot = seaborn.heatmap(poly_coef, annot=True)
plot.figure.tight_layout()
plot.figure.savefig('report.png')
