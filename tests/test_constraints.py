"""Test constraints."""
from reasoner_pydantic import Message, AttributeConstraint, Node
from strider.constraints import satisfies_attribute_constraint, enforce_constraints


def test_equal():
    """Test satisfies_attribute_constraint() with operator "=="."""
    qnode = Node.parse_obj(
        {
            "attributes": [
                {
                    "attribute_type_id": "test",
                    "value": "foo",
                },
            ],
        }
    )
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "operator": "==",
            "value": "foo",
        }
    )
    assert satisfies_attribute_constraint(qnode, constraint)
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "operator": "==",
            "value": "bar",
        }
    )
    assert not satisfies_attribute_constraint(qnode, constraint)
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "not": True,
            "operator": "==",
            "value": "bar",
        }
    )
    assert satisfies_attribute_constraint(qnode, constraint)


def test_gt():
    """Test satisfies_constraint() with opertor ">"."""
    qnode = Node.parse_obj(
        {
            "attributes": [
                {
                    "attribute_type_id": "test",
                    "value": 5,
                },
            ],
        }
    )
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "operator": ">",
            "value": 3,
        }
    )
    assert satisfies_attribute_constraint(qnode, constraint)
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "operator": ">",
            "value": 7,
        }
    )
    assert not satisfies_attribute_constraint(qnode, constraint)
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "not": True,
            "operator": ">",
            "value": 7,
        }
    )
    assert satisfies_attribute_constraint(qnode, constraint)


def test_match():
    """Test satisfies_constraint() with opertor "matches"."""
    qnode = Node.parse_obj(
        {
            "attributes": [
                {
                    "attribute_type_id": "test",
                    "value": "Mississippi",
                },
            ],
        }
    )
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "operator": "matches",
            "value": r".*(iss){2,}.*",
        }
    )
    assert satisfies_attribute_constraint(qnode, constraint)
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "operator": "matches",
            "value": r".*(iss){3,}.*",
        }
    )
    assert not satisfies_attribute_constraint(qnode, constraint)
    constraint = AttributeConstraint.parse_obj(
        {
            "name": "test",
            "id": "test",
            "not": True,
            "operator": "matches",
            "value": r".*(iss){3,}.*",
        }
    )
    assert satisfies_attribute_constraint(qnode, constraint)


def test_enforce_constraints():
    """Test enforce_constraints()."""
    message = Message.parse_obj(
        {
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
    )
    assert message == enforce_constraints(message)
    message = Message.parse_obj(
        {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "constraints": [
                            {
                                "name": "test",
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
                    },
                    "analyses": [],
                },
                {
                    "node_bindings": {
                        "n0": [
                            {"id": "b"},
                        ],
                    },
                    "analyses": [],
                },
            ],
        }
    )
    constrained = enforce_constraints(message)
    assert len(constrained.results) == 1
    assert list(constrained.knowledge_graph.nodes.keys()) == ["a"]
