import json
import time

from texttable import Texttable
from tqdm import tqdm
from reasoner_pydantic import Message

from strider.optimized_message_store import OptimizedMessageStore
from tests.helpers.utils import generate_message_parameterized

benchmarks = [
    {
        "name": "1k Results - Small",
        "msg_count": 1000,
        "params" : {
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

table = Texttable()
table.add_row(["Benchmark Name", "Output Size (MB)", "Total Time (s)"])

for b in benchmarks:
    input_messages = [
        generate_message_parameterized(**b["params"]).dict()
        for _ in range(b["msg_count"])
    ]

    start = time.time()

    print(f"Running benchmark {b['name']}")
    store = OptimizedMessageStore()
    for m in tqdm(input_messages):
        store.add_message(m)

    end = time.time()

    # Compute file size
    print("Computing final message size, this may take a while...")
    output_file_size = len(Message.parse_obj(store.get_message()).json().encode('utf-8'))

    table.add_row([b["name"], f"{output_file_size/1e6}", f"{end - start:.2f}"])

print(table.draw())
