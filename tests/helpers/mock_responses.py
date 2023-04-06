kp_response = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["MESH:D008687"]},
                "n1": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "n0n1": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:treats"],
                }
            },
        },
        "knowledge_graph": {
            "nodes": {
                "MESH:D008687": {
                    "categories": ["biolink:SmallMolecule"],
                    "name": "Metformin",
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                },
            },
            "edges": {
                "n0n1": {
                    "subject": "MESH:D008687",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:treats",
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "MESH:D008687",
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                        },
                    ],
                },
                "edge_bindings": {
                    "n0n1": [
                        {
                            "id": "n0n1",
                        },
                    ],
                },
            },
        ],
    },
}

duplicate_result_response = {
    "message": {
        "knowledge_graph": {
            "nodes": {
                "CHEBI:6801": {
                    "categories": ["biolink:SmallMolecule"],
                    "name": "Metformin",
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                },
            },
            "edges": {
                "n0n1": {
                    "subject": "CHEBI:6801",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:treats",
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:6801",
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                        },
                    ],
                },
                "edge_bindings": {
                    "n0n1": [
                        {
                            "id": "n0n1",
                        },
                    ],
                },
            },
        ],
    },
}

duplicate_result_response_2 = {
    "message": {
        "knowledge_graph": {
            "nodes": {
                "CHEBI:6801": {
                    "categories": ["biolink:SmallMolecule"],
                    "name": "Metformin",
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:DiseaseOrPhenotypicFeature",
                    ],
                    "name": "type 2 diabetes mellitus",
                },
            },
            "edges": {
                "n0n1": {
                    "subject": "CHEBI:6801",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:treats",
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:6801",
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                        },
                    ],
                },
                "edge_bindings": {
                    "n0n1": [
                        {
                            "id": "n0n1",
                        },
                    ],
                },
            },
        ],
    },
}

duplicate_result_response_different_predicate = {
    "message": {
        "knowledge_graph": {
            "nodes": {
                "CHEBI:6801": {
                    "categories": ["biolink:SmallMolecule"],
                    "name": "Metformin",
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                },
            },
            "edges": {
                "n0n1": {
                    "subject": "CHEBI:6801",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:affects",
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:6801",
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                        },
                    ],
                },
                "edge_bindings": {
                    "n0n1": [
                        {
                            "id": "n0n1",
                        },
                    ],
                },
            },
        ],
    },
}

loop_response_1 = {
    "message": {
        "knowledge_graph": {
            "nodes": {
                "CHEBI:6801": {
                    "categories": ["biolink:SmallMolecule"],
                    "name": "Metformin",
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                },
            },
            "edges": {
                "n0n1": {
                    "subject": "CHEBI:6801",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:affects",
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:6801",
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                        },
                    ],
                },
                "edge_bindings": {
                    "n0n1": [
                        {
                            "id": "n0n1",
                        },
                    ],
                },
            },
        ],
    },
}

response_with_attributes = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["MESH:D008687"]},
                "n1": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "n0n1": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:treats"],
                }
            },
        },
        "knowledge_graph": {
            "nodes": {
                "MESH:D008687": {
                    "categories": ["biolink:SmallMolecule"],
                    "name": "Metformin",
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                },
            },
            "edges": {
                "n0n1": {
                    "subject": "MESH:D008687",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:treats",
                    "attributes": [
                        {
                            "value": "infores:kp3",
                            "attribute_type_id": "biolink:knowledge_source",
                        }
                    ],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "MESH:D008687",
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                        },
                    ],
                },
                "edge_bindings": {
                    "n0n1": [
                        {
                            "id": "n0n1",
                        },
                    ],
                },
            },
        ],
    },
}
