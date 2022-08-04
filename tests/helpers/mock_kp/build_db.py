"""Data I/O."""
import re
import uuid

import aiosqlite


async def get_data_from_string(data: str):
    """Get data from string.

    Each line should be of the form:
    <CURIE>(( category <category> ))
    or
    <CURIE>-- predicate <predicate> --><CURIE>
    """
    node_pattern = r"(?P<id>[\w:]+)" r"\(\( category (?P<category>[\w:]+) \)\)"
    edge_pattern = (
        r"(?P<source>[\w:]+)"
        r"(?P<o2s><?)-- predicate (?P<predicate>[\w:]+) --(?P<s2o>>?)"
        r"(?P<target>[\w:]+)"
    )
    nodes = {}
    edges = {}
    for line in data.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.fullmatch(node_pattern, line)
        if match is not None:
            nid = match.group("id")
            nodes[nid] = {
                "id": nid,
                "category": match.group("category"),
            }

            continue

        match = re.fullmatch(edge_pattern, line)
        if match is not None:
            eid = str(uuid.uuid4())

            predicate = match.group("predicate")
            source = match.group("source")
            target = match.group("target")
            if match.group("o2s"):
                subject = target
                object = source
            else:
                subject = source
                object = target

            edges[eid] = {
                "id": eid,
                "subject": subject,
                "predicate": predicate,
                "object": object,
            }
            continue

        raise ValueError(f"Failed to parse '{line}'")
    return list(nodes.values()), list(edges.values())


async def add_data_from_string(
    connection: aiosqlite.Connection,
    data: str,
    **kwargs,
):
    """Add data to SQLite database."""
    nodes, edges = await get_data_from_string(data)
    await add_data(connection, nodes, edges)


async def add_data(
    connection: aiosqlite.Connection,
    nodes,
    edges,
):
    if nodes:
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS nodes ({0})".format(
                ", ".join([f"{val} text" for val in nodes[0]])
            )
        )
        await connection.executemany(
            "INSERT INTO nodes VALUES ({0})".format(", ".join(["?" for _ in nodes[0]])),
            [list(node.values()) for node in nodes],
        )
    if edges:
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS edges ({0})".format(
                ", ".join([f"{val} text" for val in edges[0]])
            )
        )
        await connection.executemany(
            "INSERT INTO edges VALUES ({0})".format(", ".join(["?" for _ in edges[0]])),
            [list(edge.values()) for edge in edges],
        )
    await connection.commit()
