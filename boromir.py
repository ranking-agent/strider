"""
 ______   _______  _______  _______  _______ _________ _______ 
(  ___ \ (  ___  )(  ____ )(  ___  )(       )\__   __/(  ____ )
| (   ) )| (   ) || (    )|| (   ) || () () |   ) (   | (    )|
| (__/ / | |   | || (____)|| |   | || || || |   | |   | (____)|
|  __ (  | |   | ||     __)| |   | || |(_)| |   | |   |     __)
| (  \ \ | |   | || (\ (   | |   | || |   | |   | |   | (\ (   
| )___) )| (___) || ) \ \__| (___) || )   ( |___) (___| ) \ \__
|/ \___/ (_______)|/   \__/(_______)|/     \|\_______/|/   \__/
"""                                                               
import aiostream
import asyncio
from collections import defaultdict
from collections.abc import Iterable
import copy
import httpx
import json
import logging
from reasoner_pydantic import (
    KnowledgeGraph,
    Node,
    QNode,
    Message,
    AuxiliaryGraphs,
    Result,
    Results,
    Response,
    Query,
    QueryGraph,
    Workflow,
)
from reasoner_pydantic.utils import HashableMapping
import time
from typing import Callable, List
import uuid

from strider.graph import Graph
from strider.query_planner import get_next_qedge
from strider.utils import (
    elide_curies,
    get_curies,
)

logging.basicConfig(format="[%(asctime)s: %(levelname)s/%(name)s]: %(message)s", level=logging.DEBUG)

strider_url = "https://strider.renci.org/1.4/query"
aragorn_url = "https://aragorn.renci.org/1.4/query"
# with overlay and score
num_results = 100
num_intermediate_hops = 3

message = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {
                    "ids":["NCBIGene:3815"],
                    "name": "kit"
                },
                # "n0": {
                #     "ids": ["PUBCHEM.COMPOUND:5291"],
                #     "name": "imatinib",
                # },
                "n1": {
                    "ids":["MONDO:0004979"],
                    "categories":["biolink:DiseaseOrPhenotypicFeature"],
                    "name": "asthma"
                }
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": [
                        "biolink:related_to"
                    ]
                }
            }
        },
        "knowledge_graph": {
            "nodes": {
                "MONDO:0004979": {

                }
            }
        }
    }
}

buckets = []


def get_target_node_id(qgraph: Graph) -> dict:
    """Given a qgraph, find the dangling node."""
    assert len(qgraph["edges"]) == 1
    edge = next(iter(qgraph["edges"].values()))
    assert qgraph["nodes"][edge["object"]]["ids"] and len(qgraph["nodes"][edge["object"]]["ids"]) == 1
    return edge["object"]


def build_result(result, onehop_kgraph, onehop_auxgraphs):
    """"""
    # add edge to results and kgraph

    # collect all auxiliary graph ids from results and edges
    aux_graphs = [
        aux_graph_id
        for analysis in result.analyses or []
        for aux_graph_id in analysis.support_graphs or []
    ]

    aux_graphs.extend(
        [
            aux_graph_id
            for analysis in result.analyses or []
            for _, bindings in analysis.edge_bindings.items()
            for binding in bindings
            for attribute in onehop_kgraph.edges[binding.id].attributes
            or []
            if attribute.attribute_type_id == "biolink:support_graphs"
            for aux_graph_id in attribute.value
        ]
    )

    result_auxgraph = AuxiliaryGraphs.parse_obj(
        {
            aux_graph_id: onehop_auxgraphs[aux_graph_id]
            for aux_graph_id in aux_graphs
        }
    )

    # get all edge ids from the result
    kgraph_edge_ids = [
        binding.id
        for analysis in result.analyses or []
        for _, bindings in analysis.edge_bindings.items()
        for binding in bindings
    ]

    # get all edge ids from auxiliary graphs
    kgraph_edge_ids.extend(
        [
            edge_id
            for aux_graph_id in aux_graphs
            for edge_id in result_auxgraph[aux_graph_id].edges or []
        ]
    )

    try:
        result_kgraph = KnowledgeGraph.parse_obj(
            {
                "nodes": {
                    binding.id: onehop_kgraph.nodes[binding.id]
                    for _, bindings in result.node_bindings.items()
                    for binding in bindings
                },
                "edges": {
                    edge_id: onehop_kgraph.edges[edge_id]
                    for edge_id in kgraph_edge_ids
                },
            }
        )
    except Exception as e:
        logging.error(
            f"Something went wrong making the sub-result kgraph: {e}"
        )
        raise Exception(e)
    
    return result_auxgraph, result_kgraph


def validate_kgraph(kgraph):
    for edge in kgraph["edges"].values():
        assert edge["subject"] in kgraph["nodes"]
        assert edge["object"] in kgraph["nodes"]


async def lookup(
    qgraph: Graph = None,
    target_node_id: str = None,
    target_node_curie: str = None,
    qid: str = "",
):
    """Expand from query graph node."""
    if qid:
        qid = qid + "." + str(uuid.uuid4())[:8]
    else:
        qid = str(uuid.uuid4())[:8]
    # if this is a leaf node, we're done
    if qgraph is None:
        qgraph = Graph(message["message"]["query_graph"])
    if not qgraph["edges"]:
        logging.info(f"[{qid}] Finished call stack")
        # gets sent to generate_from_result for final result merge and then yield to server.py
        yield KnowledgeGraph.parse_obj(
            {"nodes": dict(), "edges": dict()}
        ), Result.parse_obj(
            {
                "node_bindings": dict(),
                "analyses": [],
            }
        ), AuxiliaryGraphs.parse_obj(
            {}
        ), qid
        # doesn't kill generator, just don't continue on with function
        return

    try:
        # contrary to Strider, we don't find next best edge, we just grab the next one
        qedge_id, qedge = next(iter(qgraph["edges"].items()))
    except StopIteration:
        logging.error("Cannot find qedge with pinned endpoint")
        raise RuntimeError("Cannot find qedge with pinned endpoint")
    except Exception as e:
        logging.error(f"Unable to get next qedge {e}")
        raise RuntimeError("Cannot find qedge with pinned endpoint")

    logging.info(f"[{qid}] Getting results for {qedge_id}")

    onehop = {
        "nodes": {
            key: value
            for key, value in qgraph["nodes"].items()
            if key in (qedge["subject"], qedge["object"])
        },
        "edges": {qedge_id: qedge},
    }

    generators = [
        generate_from_strider(
            qgraph,
            onehop,
            target_node_id,
            target_node_curie,
            qid,
        )
    ]
    async with aiostream.stream.merge(*generators).stream() as streamer:
        async for output in streamer:
            yield output

async def generate_from_strider(
    qgraph: Graph,
    onehop_qgraph: Graph,
    target_node_id: str,
    target_node_curie: str,
    qid: str,
):
    """Generate one-hop results from Strider."""
    result_map = defaultdict(list)
    generators = []

    logging.info(
        f"[{qid}] Need to get results for: {json.dumps(elide_curies(onehop_qgraph))}"
    )
    async with httpx.AsyncClient(timeout=300) as client:
        lookup_response = await client.post(
            url=strider_url,
            json={"message": {"query_graph": onehop_qgraph}},
        )
        lookup_response.raise_for_status()
        now = time.time()
        lookup_response = lookup_response.json()
        with open("lookup_response.json", "w") as f:
            json.dump(lookup_response, f, indent=2)
        # logging.info(f"to json and dump took {time.time() - now}")

    # convert to pydantic to do result parsing
    validate_kgraph(lookup_response["message"]["knowledge_graph"])
    now = time.time()
    onehop_response = Response.parse_obj(lookup_response).message
    # logging.info(f"Pydantic parsing took {time.time() - now}")
    now = time.time()
    with open("normalized_lookup.json", "w") as f:
        json.dump(onehop_response.dict(), f, indent=2)
    # logging.info(f"Saving normalized json took {time.time() - now}")
    onehop_kgraph = onehop_response.knowledge_graph
    validate_kgraph(onehop_kgraph.dict())
    onehop_results = onehop_response.results
    onehop_auxgraphs = onehop_response.auxiliary_graphs
    logging.info(f"Got back {len(onehop_results)} results")
    # logging.info("Adding target to lookup response")
    sent_curies = get_curies(onehop_response)
    # take out any direct target relations
    now = time.time()
    results = Results.parse_obj([])
    for result in onehop_results:
        if not (
            # search for target node in result
            any(
                target_node_curie in [node_binding.id for node_binding in node_bindings]
                for node_bindings in result.node_bindings.values()
            ) and (target_node_curie not in sent_curies) # handle last hop where we actually ask for target node
        ):
            # add results for next hop
            results.append(result)
        elif (
            len(onehop_qgraph["edges"]) > 1 and any(
                target_node_curie in [node_binding.id for node_binding in node_bindings]
                for node_bindings in result.node_bindings.values()
            )
        ):
            # on hops past the first one, save any results pointing to the target node
            result_auxgraph, result_kgraph = build_result(result, onehop_kgraph, onehop_auxgraphs)
            # get intersection of result node ids and new sub qgraph
            # should be empty on last hop because the qgraph is empty
            qnode_ids = set()
            # result key becomes ex. ((n0, (MONDO:0005737,)), (n1, (RXCUI:340169,)))
            key_fcn = lambda res: tuple(
                (
                    qnode_id,
                    tuple(
                        binding.query_id if binding.query_id else binding.id
                        for binding in bindings
                    ),  # probably only one
                )
                # for cyclic queries, the qnode ids can get out of order, so we need to sort the keys
                for qnode_id, bindings in sorted(res.node_bindings.items())
                if qnode_id in qnode_ids
            )
            result_map[key_fcn(result)].append(
                (result, result_kgraph, result_auxgraph)
            )
            generators.append(
                generate_from_result(
                    # empty qgraph to end the call stack
                    {},
                    # key_fcn should be an empty tuple
                    key_fcn,
                    lambda result: result_map[key_fcn(result)],
                    result_map,
                    target_node_id,
                    target_node_curie,
                    qid,
                )
            )

    # logging.info(f"Parsed the results in {time.time() - now}")
    # put target back in and add to all result node bindings
    now = time.time()
    onehop_kgraph.nodes[target_node_curie] = Node.parse_obj(message["message"]["knowledge_graph"]["nodes"][target_node_curie])
    onehop_response.query_graph.nodes[target_node_id] = QNode.parse_obj(message["message"]["query_graph"]["nodes"][target_node_id])
    for result in results:
        result.node_bindings[target_node_id] = result.node_bindings.get(target_node_id, [{"id": target_node_curie}])
    # results was a pydantic object, now just turn it back into a dict for Aragorn
    lookup_response = Query(
        message=Message(
            query_graph=onehop_response.query_graph,
            knowledge_graph=onehop_kgraph,
            results=results,
            auxiliary_graphs=onehop_auxgraphs,
        ),
        workflow=Workflow.parse_obj([
            {"id": "overlay_connect_knodes"},
            {"id": "score"},
            {"id": "sort_results_score", "parameters": {"ascending_or_descending": "descending"}},
            {"id": "filter_results_top_n", "parameters": {"max_results": num_results}},
            {"id": "filter_kgraph_orphans"},
        ]),
    ).dict()
    validate_kgraph(lookup_response["message"]["knowledge_graph"])
    logging.info(f"Num knodes: {len(lookup_response['message']['knowledge_graph']['nodes'].keys())}")
    # logging.info(f"Got response ready for Ranker in {time.time() - now}")
    now = time.time()
    with open("lookup_response_temp.json", "w") as f:
        json.dump(lookup_response, f, indent=2)
    # logging.info(f"Saved lookup response temp in {time.time() - now}")
    logging.info(f"Sending {len(lookup_response['message']['results'])} results for scoring and filtering.")
    async with httpx.AsyncClient(timeout=300) as client:
        ranked_response = await client.post(
            url=aragorn_url,
            json=lookup_response,
        )
        ranked_response.raise_for_status()
        ranked_response = ranked_response.json()
        with open("ranked_response.json", "w") as f:
            json.dump(ranked_response, f, indent=2)

    validate_kgraph(ranked_response["message"]["knowledge_graph"])
    logging.info(f"Sending along {len(ranked_response['message']['results'])} results")
    # back into pydantic for result parsing
    now = time.time()
    onehop_response = Response.parse_obj(ranked_response).message
    # logging.info(f"Ranker response parsing took {time.time() - now}")
    now = time.time()
    onehop_kgraph = onehop_response.knowledge_graph
    onehop_results = onehop_response.results
    onehop_auxgraphs = onehop_response.auxiliary_graphs
    qedge_id = next(iter(onehop_qgraph["edges"].keys()))

    if onehop_results:
        subqgraph = copy.deepcopy(qgraph)
        # remove edge
        subqgraph["edges"].pop(qedge_id)
        # remove orphaned nodes
        subqgraph.remove_orphaned()
    else:
        logging.info(
            f"[{qid}] Ending call stack with no results"
        )
        return
    # copy subqgraph between each batch
    # before we fill it with result curies
    # this keeps the sub query graph from being modified and passing
    # extra curies into subsequent batches
    populated_subqgraph = copy.deepcopy(subqgraph)
    # clear out any existing bindings to only use the new ones we get back
    for qnode_id in onehop_qgraph["nodes"].keys():
        if qnode_id in populated_subqgraph["nodes"]:
            populated_subqgraph["nodes"][qnode_id]["ids"] = []
    for result in onehop_results:
        result_auxgraph, result_kgraph = build_result(result, onehop_kgraph, onehop_auxgraphs)

        # pin nodes
        for qnode_id, bindings in result.node_bindings.items():
            if qnode_id not in populated_subqgraph["nodes"]:
                continue
            # add curies from result into the qgraph
            populated_subqgraph["nodes"][qnode_id]["ids"] = list(
                # need to call set() to remove any duplicates
                set(
                    (populated_subqgraph["nodes"][qnode_id].get("ids") or [])
                    # use query_id (original curie) for any subclass results
                    + [binding.query_id or binding.id for binding in bindings]
                )
            )

        # get intersection of result node ids and new sub qgraph
        # should be empty on last hop because the qgraph is empty
        qnode_ids = set(populated_subqgraph["nodes"].keys()) & set(
            result.node_bindings.keys()
        )
        # result key becomes ex. ((n0, (MONDO:0005737,)), (n1, (RXCUI:340169,)))
        key_fcn = lambda res: tuple(
            (
                qnode_id,
                tuple(
                    binding.query_id if binding.query_id else binding.id
                    for binding in bindings
                ),  # probably only one
            )
            # for cyclic queries, the qnode ids can get out of order, so we need to sort the keys
            for qnode_id, bindings in sorted(res.node_bindings.items())
            if qnode_id in qnode_ids
        )
        result_map[key_fcn(result)].append(
            (result, result_kgraph, result_auxgraph)
        )

        # logging.info(
        #     f"[{qid}] put key {key_fcn(result)} in result map"
        # )

    # logging.info(f"Result parsing took {time.time() - now}")
    generators.append(
        generate_from_result(
            populated_subqgraph,
            key_fcn,
            lambda result: result_map[key_fcn(result)],
            result_map,
            target_node_id,
            target_node_curie,
            qid,
        )
    )

    async with aiostream.stream.merge(*generators).stream() as streamer:
        async for result in streamer:
            yield result

async def generate_from_result(
    qgraph,
    key_fcn,
    get_results: Callable[[dict], Iterable[tuple[dict, dict]]],
    result_map,
    target_node_id,
    target_node_curie,
    sub_qid: str,
):
    async for subkgraph, subresult, subauxgraph, qid in lookup(
        qgraph,
        target_node_id,
        target_node_curie,
        sub_qid,
    ):
        # logging.info(
        #     f"[{qid}] looking for key {key_fcn(subresult)}: {subresult.json()}"
        # )
        if not key_fcn(subresult) in result_map:
            logging.error(
                f"[{qid}] Couldn't find subresult in result map: {key_fcn(subresult)}"
            )
            logging.error(f"[{sub_qid}] Result map: {result_map.keys()}")
            logging.error(f"[{qid}] subresult from lookup: {subresult.json()}")
            raise KeyError("Subresult not found in result map")
        for result, kgraph, auxgraph in get_results(subresult):
            # combine one-hop with subquery results
            # Need to create a new result with all node bindings combined
            new_subresult = Result.parse_obj(
                {
                    "node_bindings": {
                        **subresult.node_bindings,
                        **result.node_bindings,
                    },
                    "analyses": [
                        *subresult.analyses,
                        *result.analyses,
                        # reconsider
                    ],
                }
            )
            new_subkgraph = copy.deepcopy(subkgraph)
            new_subkgraph.nodes.update(kgraph.nodes)
            new_subkgraph.edges.update(kgraph.edges)
            new_auxgraph = copy.deepcopy(subauxgraph)
            new_auxgraph.update(auxgraph)
            yield new_subkgraph, new_subresult, new_auxgraph, qid

def combine_results(query_graph, m):
    # output should have only one result, with node bindings from original query_graph
    output_result = {
        "node_bindings": {},
        "analyses": []
    }
    for qid, qnode in query_graph["nodes"].items():
        output_result["node_bindings"][qid] = [{"id": qnode["ids"][0]}]
    # we can use the aux graphs and kg from the original message as the base
    output_kg = m["knowledge_graph"]
    output_aux = m["auxiliary_graphs"]
    qedge_id = next(iter(query_graph["edges"].keys()))
    for result in m["results"]:
        for analysis in result["analyses"]:
            # create a new auxgraph for each analysis of each result (there should only be one analysis per result)
            aux_graph = {"edges":[]}
            aux_graph_id = str(uuid.uuid4())
            for edge_binding in analysis["edge_bindings"].values():
                for e in edge_binding:
                    # use edge bindings to creat the aux graph
                    aux_graph["edges"].append(e["id"])
            output_aux[aux_graph_id] = aux_graph
            # create a new kedge for this path and add the aux graph as a support graph
            edge_id = str(uuid.uuid4())
            output_kg["edges"][edge_id] = {
                "subject": "PUBCHEM.COMPOUND:5291",
                "object": "MONDO:0004979",
                "predicate": "biolink:related_to",
                "sources": [
                    {
                        "resource_id": "infores:boromir",
                        "resource_role": "primary_knowledge_source"
                    }
                ],
                "attributes": [
                    {
                        "attribute_type_id": "biolink:support_graphs",
                        "value": [aux_graph_id]
                    }
                ]
            }
            # create a new analysis with the new edge as the edge binding
            ana = {
                "resource_id": "infores:boromir",
                "edge_bindings": {
                    "e0": [
                        {
                            "id": edge_id
                        }
                ]
                },
                "support_graphs": analysis["support_graphs"],
                "score": analysis["score"]
            }
            output_result["analyses"].append(ana)

    output = {
        "message": {
            "query_graph": query_graph,
            "knowledge_graph": output_kg,
            "results": [output_result],
            "auxiliary_graphs": output_aux
        }
    }
    return output

async def main():
    unsolved_qgraph = Graph(copy.deepcopy(message["message"]["query_graph"]))
    target_node_id = get_target_node_id(copy.deepcopy(message["message"]["query_graph"]))
    target_node_curie = message["message"]["query_graph"]["nodes"][target_node_id]["ids"][0]
    final_response = {}

    # remove edge connecting source to target
    edge_id, edge = next(iter(unsolved_qgraph["edges"].items()))
    source_node_id = edge["subject"]
    unsolved_qgraph["edges"].pop(edge_id)

    # Result container to make result merging much faster
    output_results = HashableMapping[str, Result]()

    output_kgraph = KnowledgeGraph.parse_obj({"nodes": {}, "edges": {}})

    output_auxgraphs = AuxiliaryGraphs.parse_obj({})
    for hop_index in range(num_intermediate_hops):
        # add new NamedThing node into graph
        new_node_id = f"nt{hop_index}"
        unsolved_qgraph["nodes"][new_node_id] = {
            "categories": ["biolink:NamedThing"],
        }
        unsolved_qgraph["edges"][f"e{hop_index}"] = {
            "subject": source_node_id,
            "object": new_node_id,
            "predicates": ["biolink:related_to"],
        }
        source_node_id = new_node_id
    unsolved_qgraph["edges"][f"e{hop_index + 1}"] = {
        "subject": source_node_id,
        "object": target_node_id,
        "predicates": ["biolink:related_to"],
    }
    # qgraph has num_hops
    # lookup is async generator that calls Strider and Aragorn Ranker for each hop
    async for result_kgraph, result, result_auxgraph, sub_qid in lookup(
        qgraph=unsolved_qgraph,
        target_node_id=target_node_id,
        target_node_curie=target_node_curie,
    ):
        # Update the kgraph
        output_kgraph.update(result_kgraph)

        # Update the aux graphs
        output_auxgraphs.update(result_auxgraph)

        # Update the results
        # hashmap lookup is very quick
        sub_result_hash = hash(result)
        existing_result = output_results.get(sub_result_hash, None)
        if existing_result:
            # update existing result
            existing_result.update(result)
        else:
            # add new result to hashmap
            output_results[sub_result_hash] = result
    
    results = Results.parse_obj([])
    for result in output_results.values():
        # copy so result analyses don't get combined somehow
        result = copy.deepcopy(result)
        if len(result.analyses) > 1:
            result.combine_analyses_by_resource_id()
        results.append(result)

    output_query = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(message["message"]["query_graph"]),
            knowledge_graph=output_kgraph,
            results=results,
            auxiliary_graphs=output_auxgraphs,
        )
    )

    with open("penultimate_response.json", "w") as f:
        json.dump(output_query.dict(exclude_none=True), f, indent=2)


    # combine results
    merged_output_query = combine_results(
        message["message"]["query_graph"],
        output_query.message.dict(),
    )

    with open("final_response.json", "w") as f:
        json.dump(merged_output_query, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
