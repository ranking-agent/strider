"""Query graph utilities."""
import re

from bmt import Toolkit

BMT = Toolkit()


def get_subcategories(category):
    """Get sub-categories, according to the Biolink model."""
    categories = BMT.get_descendants(category, formatted=True, reflexive=True) or [
        category
    ]
    if "biolink:SmallMolecule" in categories:
        categories.append("biolink:ChemicalSubstance")
    return [category.replace("_", "") for category in categories]


def camelcase_to_snakecase(string):
    """Convert CamelCase to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()


def get_subpredicates(predicate):
    """Get sub-predicates, according to the Biolink model."""
    curies = BMT.get_descendants(predicate, formatted=True, reflexive=True) or [
        predicate
    ]
    return ["biolink:" + camelcase_to_snakecase(curie[8:]) for curie in curies]


def is_symmetric(predicate):
    """Determine whether predicate is symmetric."""
    el = BMT.get_element(predicate)
    if el is None:
        return False
    return el.symmetric


class NoAnswersException(Exception):
    """No answers to question."""


def build_conditions(**conditions):
    """Build SQL WHERE clause.

    conditions uses a format similar to this:
    https://docs.mongodb.com/manual/tutorial/query-documents/
    """
    conditions, values = zip(
        *[build_condition(key, value) for key, value in conditions.items()]
    )
    # flatten values tuples
    values = tuple(value for tup in values for value in tup)
    if len(conditions) == 1:
        return conditions[0], values
    return " AND ".join(f"({condition})" for condition in conditions), values


def build_condition(key, value):
    """Build SQL WHERE clause."""
    if key == "$or":
        conditions, values = zip(
            *[build_conditions(**alternative) for alternative in value]
        )
        # flatten values tuples
        values = tuple(value for tup in values for value in tup)
        return " OR ".join(f"({condition})" for condition in conditions), values
    predicate, values = build_predicate(value)
    return f"{key} " + predicate, values


PREDICATES = {
    "$in": "in",
    "$lt": "<",
    "$gt": ">",
    "$le": "<=",
    "$ge": ">=",
    "$eq": "==",
    "$ne": "!=",
}


def build_predicate(predicate):
    """Build predicate."""
    if not isinstance(predicate, dict):
        return f"== ?", (predicate,)
    if len(predicate) > 1:
        raise ValueError(f"Cannot parse {predicate}")
    key, value = next((key, value) for key, value in predicate.items())
    if key == "$in":
        return "in ({})".format(", ".join("?" for _ in range(len(value)))), tuple(value)
    return f"{PREDICATES[key]} ?", (value,)
