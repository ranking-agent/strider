"""Test constraints."""
from strider.constraints import satisfies_attribute_constraint, enforce_constraints


def test_equal():
    """Test satisfies_attribute_constraint() with operator "=="."""
    qnode = {
        "attributes": [
            {
                "attribute_type_id": "test",
                "value": "foo",
            },
        ],
    }
    constraint = {
        "id": "test",
        "operator": "==",
        "value": "foo",
    }
    assert satisfies_attribute_constraint(qnode, constraint)
    constraint = {
        "id": "test",
        "operator": "==",
        "value": "bar",
    }
    assert not satisfies_attribute_constraint(qnode, constraint)
    constraint = {
        "id": "test",
        "not": True,
        "operator": "==",
        "value": "bar",
    }
    assert satisfies_attribute_constraint(qnode, constraint)


def test_gt():
    """Test satisfies_constraint() with opertor ">"."""
    qnode = {
        "attributes": [
            {
                "attribute_type_id": "test",
                "value": 5,
            },
        ],
    }
    constraint = {
        "id": "test",
        "operator": ">",
        "value": 3,
    }
    assert satisfies_attribute_constraint(qnode, constraint)
    constraint = {
        "id": "test",
        "operator": ">",
        "value": 7,
    }
    assert not satisfies_attribute_constraint(qnode, constraint)
    constraint = {
        "id": "test",
        "not": True,
        "operator": ">",
        "value": 7,
    }
    assert satisfies_attribute_constraint(qnode, constraint)


def test_match():
    """Test satisfies_constraint() with opertor "matches"."""
    qnode = {
        "attributes": [
            {
                "attribute_type_id": "test",
                "value": "Mississippi",
            },
        ],
    }
    constraint = {
        "id": "test",
        "operator": "matches",
        "value": r".*(iss){2,}.*",
    }
    assert satisfies_attribute_constraint(qnode, constraint)
    constraint = {
        "id": "test",
        "operator": "matches",
        "value": r".*(iss){3,}.*",
    }
    assert not satisfies_attribute_constraint(qnode, constraint)
    constraint = {
        "id": "test",
        "not": True,
        "operator": "matches",
        "value": r".*(iss){3,}.*",
    }
    assert satisfies_attribute_constraint(qnode, constraint)


def test_enforce_constraints():
    """Test enforce_constraints()."""
    message = {
        "query_graph": {
            "nodes": {},
            "edges": {},
        },
        "knowledge_graph": {
            "nodes": {},
            "edges": {},
        },
        "results": [],
    }
    assert message == enforce_constraints(message)
    message = {
        "query_graph": {
            "nodes": {
                "n0": {
                    "constraints": [
                        {
                            "id": "test",
                            "operator": "==",
                            "value": "foo",
                        },
                    ],
                },
            },
            "edges": {},
        },
        "knowledge_graph": {
            "nodes": {
                "a": {
                    "attributes": [
                        {
                            "attribute_type_id": "test",
                            "value": "foo",
                        },
                    ],
                },
                "b": {
                    "attributes": [
                        {
                            "attribute_type_id": "test",
                            "value": "bar",
                        },
                    ],
                },
            },
            "edges": {},
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [
                        {"id": "a"},
                    ],
                }
            },
            {
                "node_bindings": {
                    "n0": [
                        {"id": "b"},
                    ],
                }
            },
        ],
    }
    constrained = enforce_constraints(message)
    assert len(constrained["results"]) == 1
    assert list(constrained["knowledge_graph"]["nodes"].keys()) == ["a"]
