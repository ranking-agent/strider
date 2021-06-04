# Execution Overview

This document is intended to give the reader a brief overview of the steps that Strider takes to execute a query. All of this information could be found by reading the code, but a high level overview can be a valuable learning tool. 

This document includes some graphs written with the [mermaid syntax](https://mermaid-js.github.io/mermaid/#/). For the best experience, please activate the Github+Mermaid extension for either [Chrome](https://chrome.google.com/webstore/detail/github-%20-mermaid/goiiopgdnkogdbjmncgedmgpoajilohe?hl=en) or [Firefox](https://addons.mozilla.org/en-US/firefox/addon/github-mermaid/?src=recommended).

## Preprocessing

Queries come in either through the `/aquery` or `/query` endpoints, defined in [server.py](strider/server.py). When queries come in they are validated by the FastAPI route. This ensures that the query graph is present and formatted correctly.

These endpoints are effectively handled in the same way. The query is assigned a randomized ID and the query graph is added to Redis. We use Redis because it allows users to retrieve query results after processing when using the `/aquery` endpoint.

After the query graph is saved to Redis, Strider initializes a StriderWorker for processing (also sometimes known as a fetcher). StriderWorker is a wrapper around a priority queue. The priority queue is used for storing results, allowing them to be processed in parallel. This is discussed in more detail later in the document. 

For now, the setup and generate_plan methods are called to create a query plan for execution. The setup method contains some logic for standardizing the query graph beyond what is done by FastAPI. For example, categories are not required on query graph nodes so missing categories are filled in with default values. 

Another standardization step is handling prefixes. Multiple identifiers (IDs) can refer to the same item with different prefixes. For example, `NCIT:C25742` and `UMLS:C0032961` both refer to the concept of pregnancy. Strider would like to be able to combine these identifiers so internally it contains a list of "preferred" prefixes so that we can choose which identifiers we would like to have. Preferred prefixes come from the biolink model, which contains an ordered list for each category. During the setup method we convert the query graph identifiers to the preferred prefixes. This is done by contacting the node normalization service.

## Query Planning

Before execution Strider generates a query plan. Query planning is a complex topic, so the details can be found in the [Query Planning Overview](PLANNING_OVERVIEW.md) document. For now a query plan can be thought of as a list of edges and which KPs need to be contacted for each one. 

There can actually be multiple plans, because for complex query graphs there can be multiple ways to traverse the graph. For example, if we have a loop it is simple to see that it might be possible to traverse in either direction. Right now we don't have any way to prioritize a plan so we just pick the first one that is generated. It is important to note that all plans start at a pinned node.

## Execution

After planning, query execution is handled by the StriderWorker (fetcher). The StriderWorker is a wrapper around a priority queue which holds results object. The motivation for this architecture is to be able to return results before the query has finished execution. In this case you can think of a completed result as a binding of all nodes to IDs:

#### Example Query Graph:

```mermaid
graph LR
        n0(( id <br/>CHEBI:6801 ))
        n0(( category <br/>biolink:ChemicalSubstance ))
        n1(( category <br/>biolink:Disease ))
        n2(( category <br/>biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        n1-- biolink:has_phenotype -->n2
```

#### Example completed result:

```mermaid
graph LR
        n0(( id <br/>CHEBI:6801 ))
        n1(( id <br/>MONDO:0005148 ))
        n2(( id <br/>HP:0004324 ))
        n1-- biolink:treated_by -->n0
        n0-- biolink:affects_abundance_of -->n2

style n0 fill:#ff9c91
style n0 stroke:#ff4733
style n1 fill:#ff9c91
style n1 stroke:#ff4733
style n2 fill:#ff9c91
style n2 stroke:#ff4733
```


Notice that the result matches the query graph, but has all of the unbound nodes converted to bound nodes. The priority queue holds these results when they are partially bound and then the query is complete when all results are converted to bound nodes.

The first step is to use the plan to insert partially bound results to the queue. These are what are used to initiate processing. A partial binding includes a node binding to the initial pinned node. So for the example above we would insert a result that looks like this:

```json
{
    "node_bindings": {
        "n0": [{
            "id": "CHEBI:6801",
            "category": "biolink:ChemicalSubstance"
        }]
    },
    "edge_bindings": {}
}
```

After this partial result is added we can start processing. Processing picks a partial result from the queue and then uses the plan to determine the next step. Then the plan is used to find which KPs should be contacted. The KPs that we contact are restricted based on the source and target category. This is because the plan may include a KP that doesn't match the current result. With the partial result above, we will only contact KPs that have a source category of `biolink:ChemicalSubstance`.

When contacting KPs we combine the information in the plan with the current ID to create a new message that is sent to the KP. This is done in the `get_kp_request_body` function. Before sending to the KP we also go through the prefix conversion process. KPs include a set of preferred ID prefixes that may not be the same as Strider's. We use the preferred prefixes of the KP to convert IDs before sending the request.

We also convert the results from the KP to Strider's preferred prefixes. This is not just for the query graph but for the knowledge graph and results list. The utilities that are used to do this can be found in the [trapi.py](strider/trapi.py) file.

After receiving and converting KP results we merge the existing results with new ones. We do our best to combine results that have matching information. The utilities for this are also in the trapi.py file. Knowledge graph nodes are combined based on the ID, and knowledge graph edges are combined if they have the same subject,predicate,object triple. Combining these results perfectly is still an active area of development so the existing implementation can be seen as a sort of heuristic.

### Postprocessing

After the queue runs out the query is finished executing. All of the results are in Redis. If the query was submitted synchronously then the results are pulled from Redis and set to expire. For sync queries the results cannot be retrieved again.

For async queries, there is a `/query_result` endpoint that allows retrieval of results using the associated ID. There is a setting in [config.py](strider/config.py) that controls how long results are saved in Redis for. Every time results are retrieved the expiration is "reset". This is a convenience that allows the client to keep the results saved in Strider for as long as necessary by simply repeating the retrieval. 

