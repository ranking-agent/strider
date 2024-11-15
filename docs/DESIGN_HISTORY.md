# Design decision history

### ca. March 2020

components:
* Python web server
* Python worker process(es)
* RabbitMQ message broker/queue
* Redis metadata storage and KP request cache (temporary)
* Neo4j data storage/processing (temporary)
* SQLite result storage (long-term)

walkthrough:
1. Python web server receives query
1. ... computes query plan, store in Redis
1. ... enqueues "job" in RabbitMQ
1. Python worker dequeues job
1. ... gets query plan from Redis
1. ... gets one-hop results from KPs, adds to Neo4j
1. ... queries Neo4j for (partial) results
1. ... stores (partial) results in SQLite
1. ... enqueues a job for each incomplete result in RabbitMQ, if not cached

pros:
* no wasted KP effort
* can prioritize jobs easily
* Python uses very little CPU
* delivers partial results to client

cons:
* Neo4j uses lots of resources, frequently falls over
* rather complicated (see "components")

### ca. June 2020

changes:
* move RabbitMQ responsibilities into Python
  * use asyncio.PriorityQueue
* move Redis responsibilities into Python
  * hold query plan in Python
  * hold KP request cache in Python
* combine Python web server and workers in a single process (with asyncio)

pros:
* simplification

### ca. January 2021

changes:
* move Neo4j responsibilities into Python
  * partial results are retained as local variables in 

pros:
* simplification
* speed and robustness - Neo4j was bad

### ca. January 2021

changes:
* move SQLite responsibilities into Python/Redis
  * Use Redis to store results
* store logs in Redis, too 

pros:
* can return logs to client
* no more SQLite

_at this point_:

components:
* Python web server/worker
* Redis kgraph, result, and log store

walkthrough:
1. Python web server receives query
1. ... computes query plan
1. ... spawns worker "thread"
1. ... enqueues partial result
1. Python worker (thread) dequeues partial result
1. ... gets one-hop results from KPs, combines with existing partial results
1. ... enqueues a job for each new incomplete result, if not cached

### ca. August 2021

changes:
* use trapi-throttle**
* swap pre-planning and partial-result queue for recursion/generator-based traversal
  * no longer storing kgraph/results in Redis
* streamlined logging

pros:
* throttling KP requests
* big performance gains
  * batched 3rd-party requests
  * many fewer asyncio threads
* much easier to track queries
* on-line planning can be more efficient

### ca. September 2021

changes:
* internalize trapi-throttle

pros:
* improved developer experience

_at this point_:

components:
* Python web server/worker
* Redis log store

walkthrough:
1. Python web server receives query
1. ... splits query into a) one-hop and b) the rest
1. ... gets results for (a) from KPs
1. ... gets results for (b) recursively
1. ... combines results from (a) and (b)

### ca. October 2022

changes:
* remove redis and use native python logging for logs

pros:
* redis was crashing under heavy load (multiple creative mode)

components:
* Python web server/worker

### November 2024

components:
* Python web server/workers: handles all incoming API requests
* redis: stores cache of kp-registry as well as all one-hop KP requests

external services (outside KPs):
* Node Normalizer
* OTEL/Jaeger: web API tracing for full query profiles
