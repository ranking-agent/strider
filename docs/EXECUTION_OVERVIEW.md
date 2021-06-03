# Execution Overview

This document is intended to give the reader a brief overview of the steps that Strider takes to execute a query. All of this information could be found by reading the code, but a high level overview can be a valuable learning tool. 

## Preprocessing

Queries come in either through the `/aquery` or `/query` endpoints, defined in [server.py](strider/server.py). When queries come in they are validated by the FastAPI route. This ensures that the query graph is present and formatted correctly.

These endpoints are effectively handled in the same way. The query is assigned a randomized ID and the query graph is added to Redis. We use Redis because it allows users to retrieve query results after processing when using the `/aquery` endpoint.

After the query graph is saved to Redis, Strider initializes a StriderWorker for processing (also sometimes known as a fetcher). StriderWorker is a wrapper around a priority queue. The priority queue is used for storing results, allowing them to be processed in parallel. This is discussed in more detail later in the document. 

For now, the setup and generate_plan methods are called to create a query plan for execution. The setup method contains some logic for standardizing the query graph beyond what is done by FastAPI. For example, categories are not required on query graph nodes so missing categories are filled in with default values. 

Another standardization step is handling prefixes. Multiple identifiers (IDs) can refer to the same item with different prefixes. For example, `NCIT:C25742` and `UMLS:C0032961` both refer to the concept of pregnancy. Strider would like to be able to combine these identifiers so internally it contains a list of "preferred" prefixes so that we can choose which identifiers we would like to have. Preferred prefixes come from the biolink model, which contains an ordered list for each category. During the setup method we convert the query graph identifiers to the preferred prefixes. This is done by contacting the node normalization service.



## Query Planning

Before execution Strider generates a query plan. The details of what is included in the query plan can be found in the [Query Plan Overview](PLANNING_OVERVIEW.md). 
