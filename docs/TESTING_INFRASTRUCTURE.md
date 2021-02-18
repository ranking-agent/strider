# Unit Testing Infrastructure Overview



## Introduction & Goals

ARAs are complex pieces of software. If you ask a programmer how to ensure correctness with complex software, most of them will give some version of testing as an answer. It is clear that testing can be a valuable tool when developing an ARA. It's less clear *how* to implement effective testing. The main challenge is that ARAs make calls to a variety of external tools that can behave (or misbehave) in a variety of ways. When Strider receives a query it contacts the following external tools:

- KP Registry to find KPs available to solve particular edges
- Node Normalizer to convert curies between formats
- Individual KPs to solve one-hop steps of a given query

Comprehensive testing requires that we can control how these services respond. During normal operation, these services could respond with 500 errors, JSON error codes, empty lists, or null values. One of the goals of the testing infrastructure is to verify that Strider gracefully handles these cases, even if that means submitting a log message and returning!

A good example of this is how Strider handles node normalizer responses. One of the ways we use the node normalizer is to find synonyms. Using synonyms when contacting KPs helps us get more responses back. Contacting the node normalizer is helpful, but not necessary. Using our testing framework, we were able to write tests to ensure that when the node normalizer is unavailable we still return results but also include a warning message in the response. 

## Architecture Overview

A common testing pattern for large pieces of software is to build integration tests. This would involve running Strider, the KP Registry, and individual KPs in separate processes. There ar e drawbacks to this approach that make it difficult:

- Testing data is completely separated from the tests. This can make the code brittle - changing data for one test can affect all of the tests.
- Testing requires a networking infrastructure. This means there is additional tooling that must be present to run tests. 

Our infrastructure uses a feature of Python's `httpcore` library to intercept external HTTP calls and route them to internal handlers. All of this takes place within one Python process. This eliminates the need for networking infrastructure and makes the tests less like integration tests and more like unit tests.

## Networking Overlay (ASGIAR)

The code for simulating external services is packaged in the [ASGIAR Repository](https://github.com/patrickkwang/asgiar). This allows overlaying an ASGI appliction to intercept HTTP requests. ASGI is the successor to WSGI - a standardized Python web server interface. Many standard frameworks implement ASGI including FastAPI, Starlette, and Django. This means any application written using any of these frameworks can "plug in" to ASGIAR and handle web requests.

The interface for using ASGIAR uses the standard Python context handler. Running a test with a custom KP is as simple as:

```python
from asgiar import ASGIAR
from custom_kp import app as custom_kp_app

async def my_custom_kp_test():
    with ASGIAR(custom_kp_app, host="kp"):
        async with httpx.AsyncClient() as client:
            response = await client.get("http://kp/test")
        assert response.status_code == 200
```

## Custom Services

The second key piece of our infrastructure is mocking out external services, in particular the Node Normalizer, KP Registry, and KPs. These services live in separate repositories and can be installed as Python packages with pip. This allows us to use a `requirements.txt` file for testing to pull in these dependencies. The services can be found at the following repositories:

- KP Registry: [https://github.com/ranking-agent/kp\_registry](https://github.com/ranking-agent/kp_registry)
- Simple KP: [https://github.com/TranslatorSRI/simple-kp](https://github.com/TranslatorSRI/simple-kp)
- Node Normalizer: Stored directly in the Strider repository [https://github.com/ranking-agent/strider/blob/master/tests/helpers/normalizer.py](https://github.com/ranking-agent/strider/blob/master/tests/helpers/normalizer.py)

All of these services are built using FastAPI to maintain compatibility with ASGIAR. Each of them can be initialized with custom data during creation.


## Utilities

So far, we have seen a set of utilities that allow for very powerful testing all within a single Python process. However, implementing these tools as-is for tests is cumbersome. Luckily we can use helper functions that provide a clean interface for writing readable tests.

### Data Description

After many iterations of refining test structure, we decided that the cleanest way to specify test data was directly in the code. Experiments with separate test data files showed that separating the test data into JSON files made it harder to understand the tests. The main issue with writing tests in the code is that JSON gets very verbose very quickly:

```json
{
	"nodes": {
		"n0": {
			"id": "MONDO:0005148"
		},
		"n1": {
			"category" : "biolink:Drug"
		}
	},
	"edges": {
		"e01" : {
			"subject" : "n0",
			"object" : "n1",
			"predicate" : "biolink:treated_by"
		}
	}
}
```

Seven lines to specify a query graph with a single node isn't great. Most of our tests need query graphs, KP data, and Normalizer data as well. Our solution was to write helper functions that allow specifying data as a string that gets converted to JSON:

```python
query_graph = query_graph_from_string(
		"""
        n0(( id MONDO:0005148 ))
        n1(( category biolink:Drug ))
        n0-- biolink:treated_by -->n1
		"""
)
```

As a bonus, this format is compatible with the [Mermaid](https://mermaid-js.github.io/mermaid/#/) markup language. This means that we can use existing visualization utilities to display query graphs for our tests, such as the [Mermaid Live Editor](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggVERcbiAgICAgICAgbjAoKCBpZCBNT05ETzowMDA1MTQ4ICkpXG4gICAgICAgIG4xKCggY2F0ZWdvcnkgYmlvbGluazpEcnVnICkpXG4gICAgICAgIG4wLS0gYmlvbGluazp0cmVhdGVkX2J5IC0tPm4xXG5cdFx0IiwibWVybWFpZCI6e30sInVwZGF0ZUVkaXRvciI6ZmFsc2V9). This is a great tool for debugging tests quickly and efficiently.

There are similar utilities present in the code for KP data (`kps_from_string`) as well as normalizer data (`normalizer_data_from_string`).

### Wrappers

The other 

## Conclusion & Future Improvements

