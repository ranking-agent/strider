"""Mock KP responses."""

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
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                    "attributes": [],
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
                    "attributes": [],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "MESH:D008687",
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
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
                                    "attributes": [],
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
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                    "attributes": [],
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
                    "attributes": [],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:6801",
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
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
                                    "attributes": [],
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
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:DiseaseOrPhenotypicFeature",
                    ],
                    "name": "type 2 diabetes mellitus",
                    "attributes": [],
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
                    "attributes": [],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:6801",
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
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
                                    "attributes": [],
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
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                    "attributes": [],
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
                    "attributes": [],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:6801",
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
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
                                    "attributes": [],
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
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                    "attributes": [],
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
                    "attributes": [],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:6801",
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
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
                                    "attributes": [],
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
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                    "attributes": [],
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
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
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
                                    "attributes": [],
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
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                    "attributes": [],
                },
                "MESH:D014867": {
                    "categories": [
                        "biolink:ChemicalEntity",
                    ],
                    "name": "Water",
                    "attributes": [],
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
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MESH:D008687",
                            "attributes": [],
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
                                    "attributes": [],
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
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MESH:D014867",
                            "attributes": [],
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
                                    "attributes": [],
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
                "attributes": [],
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
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "type 2 diabetes mellitus",
                    "attributes": [],
                },
                "MESH:D014867": {
                    "categories": [
                        "biolink:SmallMolecule",
                    ],
                    "attributes": [],
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
                    "attributes": [],
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
                    "attributes": [],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "MESH:D000588",
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
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
                                    "attributes": [],
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
                            "attributes": [],
                        },
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
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
                                    "attributes": [],
                                },
                            ],
                        },
                    }
                ],
            },
        ],
    },
}

response_with_pinned_node_subclasses = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["MONDO:0005011"]},
                "n1": {"categories": ["biolink:NamedThing"]},
            },
            "edges": {
                "n0n1": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:related_to"],
                }
            },
        },
        "knowledge_graph": {
            "nodes": {
                "PUBCHEM.COMPOUND:2723601": {
                    "categories": ["biolink:SmallMolecule"],
                    "name": "Thioguanine",
                    "attributes": [],
                },
                "MONDO:0005011": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "Crohns",
                    "attributes": [],
                },
                "UMLS:C0156146": {
                    "categories": [
                        "biolink:Disease",
                    ],
                    "name": "Crohn's disease of small intestine",
                    "attributes": [],
                },
                "MONDO:0021074": {
                    "name": "precancerous condition",
                    "categories": [
                        "biolink:BiologicalEntity",
                        "biolink:NamedThing",
                        "biolink:ThingWithTaxon",
                        "biolink:Disease",
                        "biolink:DiseaseOrPhenotypicFeature",
                    ],
                    "attributes": [],
                },
            },
            "edges": {
                "e0": {
                    "predicate": "biolink:subclass_of",
                    "sources": [
                        {
                            "resource_id": "infores:kp1",
                            "resource_role": "primary_knowledge_source",
                            "upstream_resource_ids": None,
                        },
                        {
                            "resource_id": "infores:kp1",
                            "resource_role": "aggregator_knowledge_source",
                            "upstream_resource_ids": ["infores:kp1"],
                        },
                    ],
                    "subject": "UMLS:C0156146",
                    "attributes": [],
                    "object": "MONDO:0021074",
                },
                "e1": {
                    "predicate": "biolink:subclass_of",
                    "sources": [
                        {
                            "resource_id": "infores:kp1",
                            "resource_role": "primary_knowledge_source",
                            "upstream_resource_ids": None,
                        },
                        {
                            "resource_id": "infores:kp1",
                            "resource_role": "aggregator_knowledge_source",
                            "upstream_resource_ids": ["infores:kp1"],
                        },
                    ],
                    "subject": "UMLS:C0156146",
                    "attributes": [],
                    "object": "MONDO:0005011",
                },
            },
        },
        "results": [
            {
                "analyses": [
                    {
                        "edge_bindings": {
                            "hop1": [
                                {
                                    "id": "e0",
                                    "attributes": [],
                                }
                            ]
                        },
                        "resource_id": "infores:kp1",
                    }
                ],
                "node_bindings": {
                    "n1": [
                        {
                            "id": "MONDO:0021074",
                            "attributes": [],
                        }
                    ],
                    "n0": [
                        {
                            "id": "UMLS:C0156146",
                            "query_id": "MONDO:0005011",
                            "attributes": [],
                        }
                    ],
                },
            },
            {
                "analyses": [
                    {
                        "edge_bindings": {
                            "hop1": [
                                {
                                    "id": "e1",
                                    "attributes": [],
                                }
                            ]
                        },
                        "resource_id": "infores:kp1",
                    }
                ],
                "node_bindings": {
                    "n1": [
                        {
                            "id": "UMLS:C0156146",
                            "attributes": [],
                        }
                    ],
                    "n0": [
                        {
                            "id": "MONDO:0005011",
                            "attributes": [],
                        }
                    ],
                },
            },
        ],
    },
}

disambiguation_response = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "n0n1": {
                    "subject": "n0",
                    "predicate": "biolink:treats",
                    "object": "n1",
                },
            },
        },
        "knowledge_graph": {
            "nodes": {
                "CHEBI:6801": {
                    "categories": ["biolink:NamedThing"],
                    "name": "metformin",
                    "attributes": [],
                },
                "CHEBI:XXX": {
                    "categories": ["biolink:NamedThing"],
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": ["biolink:NamedThing"],
                    "attributes": [
                        {
                            "attribute_type_id": "test_constraint",
                            "value": "foo",
                        },
                    ],
                },
            },
            "edges": {
                "n0n1": {
                    "subject": "CHEBI:XXX",
                    "predicate": "biolink:treats",
                    "object": "MONDO:0005148",
                    "sources": [
                        {
                            "resource_id": "kp1",
                            "resource_role": "primary_knowledge_source",
                        },
                    ],
                    "attributes": [],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:XXX",
                            "query_id": "CHEBI:6801",
                            "attributes": [],
                        }
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
                        }
                    ],
                },
                "analyses": [
                    {
                        "resource_id": "infores:kp1",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                    "attributes": [],
                                }
                            ],
                        },
                    }
                ],
            },
        ],
    }
}

unbatching_response = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "n0n1": {
                    "subject": "n0",
                    "predicate": "biolink:treats",
                    "object": "n1",
                },
            },
        },
        "knowledge_graph": {
            "nodes": {
                "CHEBI:6801": {
                    "categories": ["biolink:NamedThing"],
                    "name": "metformin",
                    "attributes": [],
                },
                "CHEBI:XXX": {
                    "categories": ["biolink:NamedThing"],
                    "attributes": [],
                },
                "MONDO:0005148": {
                    "categories": ["biolink:NamedThing"],
                    "attributes": [
                        {
                            "attribute_type_id": "test_constraint",
                            "value": "foo",
                        },
                    ],
                },
            },
            "edges": {
                "n0n1": {
                    "subject": "CHEBI:XXX",
                    "predicate": "biolink:treats",
                    "object": "MONDO:0005148",
                    "sources": [
                        {
                            "resource_id": "kp1",
                            "resource_role": "primary_knowledge_source",
                        },
                    ],
                    "attributes": [],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {
                            "id": "CHEBI:XXX",
                            "query_id": "CHEBI:6801",
                            "attributes": [],
                        }
                    ],
                    "n1": [
                        {
                            "id": "MONDO:0005148",
                            "attributes": [],
                        }
                    ],
                },
                "analyses": [
                    {
                        "resource_id": "infores:kp1",
                        "edge_bindings": {
                            "n0n1": [
                                {
                                    "id": "n0n1",
                                    "attributes": [],
                                }
                            ],
                        },
                    }
                ],
            },
        ],
    }
}
