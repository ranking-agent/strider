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
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "n0n1": {
                    "subject": "n0",
                    "predicate": "biolink:related_to",
                    "object": "n1",
                },
            },
        },
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
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "n0n1": {
                    "subject": "n0",
                    "predicate": "biolink:related_to",
                    "object": "n1",
                },
            },
        },
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
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "n0n1": {
                    "subject": "n0",
                    "predicate": "biolink:related_to",
                    "object": "n1",
                },
            },
        },
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
                "n0": {"ids": ["MONDO:0005148"]},
                "n1": {"categories": ["biolink:ChemicalEntity"]},
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
                    "subject": "MONDO:0005148",
                    "object": "MESH:D008687",
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
                "extra_edge_1": {
                    "subject": "MESH:D008687",
                    "object": "MESH:D014867",
                    "predicate": "biolink:subclass_of",
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
                            "id": "MONDO:0005148",
                        },
                    ],
                    "n1": [
                        {
                            "id": "MESH:D008687",
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
                    },
                ],
            },
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "MONDO:0005148",
                        },
                    ],
                    "n1": [
                        {
                            "id": "MESH:D014867",
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
        },
    },
}

blocked_response = {
    "message": {
        "query_graph": {
            "nodes": {
                "n1": {"ids": ["MONDO:0005148"]},
                "n0": {"categories": ["biolink:SmallMolecule"]},
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
                "MESH:D000588": {
                    "categories": ["biolink:SmallMolecule"],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                },
                "MESH:D014867": {
                    "categories": [
                        "biolink:SmallMolecule",
                    ],
                },
            },
            "edges": {
                "e0": {
                    "subject": "MESH:D000588",
                    "object": "MONDO:0005148",
                    "predicate": "biolink:treats",
                    "sources": [
                        {
                            "resource_id": "infores:kp0",
                            "resource_role": "primary_knowledge_source",
                        }
                    ],
                },
                "e1": {
                    "subject": "MONDO:0005148",
                    "object": "MESH:D008687",
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
                            "id": "MESH:D000588",
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
                                    "id": "e0",
                                },
                            ],
                        },
                    }
                ],
            },
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "MESH:D014867",
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
                                    "id": "e1",
                                },
                            ],
                        },
                    }
                ],
            },
        ],
    },
}
