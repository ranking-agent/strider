import json
import time

from tqdm import tqdm

from tests.helpers.utils import generate_message_parameterized
from strider.trapi import merge_messages

RANDOM_SEED = 42

benchmarks = [
    {
        "name": "1k Results - Small",
        "msg_count": 1000,
        "params" : {
            "random_seed" : RANDOM_SEED,
            "kg_node_count": 5,
            "kg_node_categories_count": 5,
            "kg_edge_count": 5,
            "kg_attribute_count": 10,
            "result_count": 1,
            "result_attribute_count": 10,
            "attribute_value_size": 1,
            "attribute_subattribute_count": 0,
        }
    },
    {
        "name": "100 Results - Large",
        "msg_count": 100,
        "params" : {
            "random_seed" : RANDOM_SEED,
            "kg_node_count": 100,
            "kg_edge_count": 200,
            "kg_node_categories_count": 5,
            "kg_attribute_count": 10,
            "result_count": 1,
            "result_attribute_count": 1,
            "attribute_value_size": 1,
            "attribute_subattribute_count": 0,
        }
    },
    {
        "name": "1k Results - Attribute Heavy",
        "msg_count": 1000,
        "params" : {
            "random_seed" : RANDOM_SEED,
            "kg_node_count": 5,
            "kg_node_categories_count": 5,
            "kg_edge_count": 5,
            "kg_attribute_count": 100,
            "result_count": 1,
            "result_attribute_count": 10,
            "attribute_value_size": 100,
            "attribute_subattribute_count": 10,
        }
    },
]

table = "\n\n"

table += "          Benchmark Name            |  Output Size (MB)  |  Total Time (s)\n"
table += "--------------------------------------------------------------------------\n"

for b in benchmarks:
    input_messages = [
        generate_message_parameterized(**b["params"]).dict()
        for _ in range(b["msg_count"])
    ]

    start = time.time()

    combined_msg = {
        "query_graph" : {"nodes" : {}, "edges" : {}},
        "knowledge_graph" : {"nodes" : {}, "edges" : {}},
        "results" : []
    }

    print(f"Running benchmark {b['name']}")
    for m in tqdm(input_messages):
        combined_msg = merge_messages([combined_msg, m])

    end = time.time()

    # Compute file size
    print("Computing final message size, this may take a while...")
    output_file_size = len(json.dumps(combined_msg).encode('utf-8'))

    table += f"  {b['name'].center(32)}  |  {output_file_size/1e6:16}  |  {end - start:14.2f}\n"

print(table)
