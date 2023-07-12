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
                    "sources": [
                        {
                            "resource_id": "infores:kp0",
                            "resource_role": "primary_knowledge_source",
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
                "analyses": [
                    {
                        "resource_id": "kp0",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                },
                            ],
                        },
                    }
                ],
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
                    "sources": [
                        {
                            "resource_id": "infores:kp0",
                            "resource_role": "primary_knowledge_source",
                        },
                    ],
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
                "analyses": [
                    {
                        "resource_id": "kp0",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                },
                            ],
                        },
                    }
                ],
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
                    "sources": [
                        {
                            "resource_id": "infores:kp0",
                            "resource_role": "primary_knowledge_source",
                        },
                    ],
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
                "analyses": [
                    {
                        "resource_id": "kp0",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                },
                            ],
                        },
                    }
                ],
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
                    "sources": [
                        {
                            "resource_id": "infores:kp1",
                            "resource_role": "primary_knowledge_source",
                        },
                    ],
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
                "analyses": [
                    {
                        "resource_id": "kp1",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                },
                            ],
                        },
                    }
                ],
            }
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
                    "sources": [
                        {
                            "resource_id": "infores:kp0",
                            "resource_role": "primary_knowledge_source",
                        },
                    ],
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
                "analyses": [
                    {
                        "resource_id": "kp0",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                },
                            ],
                        },
                    }
                ],
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
                        },
                    ],
                    "sources": [
                        {
                            "resource_id": "infores:kp3",
                            "resource_role": "primary_knowledge_source",
                        },
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
                "analyses": [
                    {
                        "resource_id": "kp3",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                },
                            ],
                        },
                    }
                ],
            },
        ],
    },
}

response_with_aux_graphs = {
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
                "MESH:D014867": {
                    "categories": [
                        "biolink:ChemicalEntity",
                    ],
                    "name": "Water",
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
                        },
                        {
                            "attribute_type_id": "biolink:support_graphs",
                            "value": [
                                "2",
                            ],
                        },
                    ],
                    "sources": [
                        {
                            "resource_id": "infores:kp3",
                            "resource_role": "primary_knowledge_source",
                        },
                    ],
                },
                "extra_edge_1": {
                    "subject": "MESH:D014867",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:treats",
                    "attributes": [
                        {
                            "value": "infores:kp3",
                            "attribute_type_id": "biolink:knowledge_source",
                        },
                    ],
                    "sources": [
                        {
                            "resource_id": "infores:kp3",
                            "resource_role": "primary_knowledge_source",
                        },
                    ],
                },
                "extra_edge_2": {
                    "subject": "MESH:D014867",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:ameliorates",
                    "attributes": [
                        {
                            "value": "infores:kp3",
                            "attribute_type_id": "biolink:knowledge_source",
                        },
                    ],
                    "sources": [
                        {
                            "resource_id": "infores:kp3",
                            "resource_role": "primary_knowledge_source",
                        },
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
                "analyses": [
                    {
                        "resource_id": "kp3",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                },
                            ],
                        },
                        "support_graphs": [
                            "1",
                        ],
                    },
                ],
            },
        ],
        "auxiliary_graphs": {
            "1": {
                "edges": [
                    "extra_edge_1",
                ],
            },
            "2": {
                "edges": [
                    "extra_edge_2",
                ],
            },
        },
    },
}
