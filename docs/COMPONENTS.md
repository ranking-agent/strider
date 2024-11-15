# Strider Architecture

## Entrypoints

* `/query` (`server.py`) - synchronous TRAPI endpoint
  Does a little workflow stuff, but mostly calls `lookup()`
* `/asyncquery` (`server.py`) - asynchronous TRAPI endpoint
  1. Returns immediately
  2. Calls `lookup()`
  3. POSTs response to callback url

## Modules

* `caching.py` contains some caching utilities - primarily decorators for applying a cache or a locking cache to an asynchronous function
* `config.py` defines Strider settings using [Pydantic settings management](https://pydantic-docs.helpmanual.io/usage/settings/)
* `constraints.py` handles evaluating and enforcing qnode/qedge constraints.
* `fetcher.py` handles the coordination of one-hop subqueries to answer an arbitrary graph query.
* `graph.py` defines a dict extension with a couple of useful utilities for exploring TRAPI-style graphs
* `knowledge_provider.py` is a class wrapper for each KP, does biolink conversions and all pre/post processing including filtering
* `logger.py` set up the server logger
* `mcq.py` basic utility functions specifically for MCQ(MultiCurie Query)/Set Input Queries
* `node_sets.py` single function for collapsing node sets
* `normalizer.py` defines a Python client for the node normalizer service: https://github.com/TranslatorSRI/NodeNormalization, https://nodenormalization-sri.renci.org/docs
* `openapi.py` defines the TRAPI subclass of FastAPI to add the common TRAPI elements to the OpenAPI schema
* `profiler.py` handles request profiler
* `query_planner.py` contains tools for planning query graph traversals
* `server.py` builds the [FastAPI](https://fastapi.tiangolo.com/) server and endpoints
* `throttle_utils.py` contains utilities for exploring and manipulating TRAPI messages
* `throttle.py` handles batching and throttling request to KPs
* `trapi.py` defines utilities for TRAPI messages, including normalizing, merging, and result filtering
* `traversal.py` contains code for verifying that a query graph can be solved with the KPs available (traversable)
* `utils.py` :\ a whole bunch of random stuff, some of it important

## Important functions

* `Fetcher.lookup(qgraph)` (`fetcher.py`) generates (subkgraph, subresult) pairs
  1. Gets the next qedge to traverse and generates the correponding a one-hop query.
  2. Passes it to each KP that can solve (`generate_from_kp()`).

* `Fetcher.generate_from_kp(qgraph, onehop_qgraph, kp)` (`fetcher.py`) generates (subkgraph, subresult) pairs
  1. Sends one-hop query to KP. Enforces any qnode/qedge constraints afterwards.
  2. Constructs new qgraph from original by removing the traversed qedge.
  3. Separates results into batches of size at most X (now 1 million).
  4. Passes each batch to `generate_from_results()` along with a result map/function that points back to linked subresults.

* `Fetcher.generate_from_results(qgraph, get_results)` (`fetcher.py`) generates (subkgraph, subresult) pairs
  1. Calls `lookup(qgraph)` and stitches the results with back-linked subresults from `get_results()`.

`lookup()`, `generate_from_kp()`, and `generate_from_results` form a recursion such that qgraphs can be solved by extracting one-hop sub-queries and joining the results with the solution to the remainder.
To stitch the sub-results together, we have separated them out, even though every 3rd-party (KP) call works on batches of such one-hop bits.

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│                                                                                             │
│    ┌──────────┐            ┌────────────────────┐            ┌────────────────────────┐     │
│    │          │  fanout    │                    ├┐  fanout   │                        ├┐    │
└───►│  lookup  ├───────────►│  generate_from_kp  │┼──────────►│  generate_from_result  │┼────┘
     │          │            │                    ││           │                        ││
     └──────────┘            └┬───────────────────┘│           └┬───────────────────────┘│
                              └────────────────────┘            └────────────────────────┘

                                      x KPs                              x results
```
* `ThrottledServer.process_batch()` (`throttle.py`) iteratively reads from an input request queue and writes to the appropriate request queues
  1. Receives a number of requests
  2. Identifies a subset of the available requests that are merge-able, re-queues the rest
  3. Constructs batched request
  4. Preprocesses request (mapping CURIES, mostly)
  5. Sends request
  6. Re-queues sub-requests if server rejects due to rate limiting (status 429)
  7. Validates w.r.t. TRAPI and post-processes response (normalizing CURIEs, mostly)
  8. Splits (un-batches) response into provided response queues

* `ThrottledServer.query(qgraph)` (`throttle.py`) returns a TRAPI response
  This provides a synchronous interface to throttling/batching (via `process_batch()`).

A `ThrottledServer` is set up upon query initiation for each KP, and manages throttling and batching for that KP for the query lifetime.

* `Normalizer.map_curie(curie, prefixes)` returns a list of mapped CURIEs according to the preferred identifier sets
  1. Gets the preferred prefixes for the node's categories
  2. Gets all CURIEs starting with the most-preferred prefix available in the synset

* `KnowledgeProvider.map_prefixes(message, prefixes)` returns a TRAPI message with CURIEs mapped to the preferred identifier sets
  1. Gets all CURIEs from the input message
  2. Finds categories and synonyms for CURIEs
  3. Gets CURIE map using `Normalizer.map_curie()`
  4. Applies CURIE map to message

## Libraries

* [reasoner-pydantic](https://github.com/TranslatorSRI/reasoner-pydantic) - Pydantic models reflecting the TRAPI components
