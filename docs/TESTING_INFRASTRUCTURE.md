# Unit Testing Infrastructure Overview



## Introduction & Goals

An ARA is a complex pieces of software. One of the most important tools for building complex software is testing. It's less clear *how* to implement effective testing for an ARA. The main challenge is that ARAs make calls to external tools that can behave (or misbehave) in a variety of ways. When Strider receives a query it contacts the following external tools:

- KP Registry to find KPs available to solve particular edges
- Node Normalizer to convert curies between formats
- Individual KPs to solve one-hop steps of a given query

Comprehensive testing requires that we can control how these services respond. During normal operation, these services could respond with 500 errors, JSON error codes, empty lists, or null values. One of the goals of the testing infrastructure is to verify that Strider gracefully handles these cases, even if that means submitting a log message and returning.

A good example of this is how Strider handles node normalizer responses. One of the ways we use the node normalizer is to find synonyms. Using synonyms when contacting KPs causes them to return more results. Contacting the node normalizer is helpful, but not necessary. It is important for us to verify that Strider gracefully handles when the normalizer is unavailable. Specifically, we would like to ensure that a valid ressponse is returned, with results, and including a log message saying that the normalizer was unavailable. One of the high level goals of the testing framework is to support testing these types of use cases.

## Architecture Overview

A common testing pattern for large pieces of software is to build integration tests. This would involve running Strider, the KP Registry, and individual KPs in separate processes. There ar e drawbacks to this approach that make it difficult. One main drawback is that testing requires a networking infrastructure. This means there is additional tooling that must be present to run tests.

Our infrastructure uses a feature of Python's `httpcore` library to intercept external HTTP calls and route them to internal handlers. All of this takes place within one Python process. This eliminates the need for networking infrastructure and makes the tests less like integration tests and more like unit tests.

## Networking Overlay (ASGIAR)

The code for simulating external services is packaged in the [ASGIAR Repository](https://github.com/patrickkwang/asgiar). This allows overlaying an ASGI appliction to intercept HTTP requests. ASGI is the successor to WSGI - a standardized Python web server interface. Many frameworks implement ASGI including FastAPI, Starlette, and Django. This means any application written using any of these frameworks can "plug in" to ASGIAR and handle web requests.

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

In this test, the call to `client.get` will be handled by the `/test` endpoint of the custom\_kp\_app.

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

Sixteen lines to specify a query graph with a single edge isn't great. Most of our tests need a query graph, KP data, and Normalizer data. Our solution was to write helper functions that allow specifying data as a string that gets converted to JSON:

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

### Decorators

The other utility function helps remove the issues of nested context providers. Running ASGIAR code with multiple hosts requires nested `with` statements and quickly begins to look like [Javascript from 2012](http://callbackhell.com/):

```python
with ASGIAR(custom_kp_1, host="kp1"):
    with ASGIAR(custom_kp_2, host="kp2"):
		with ASGIAR(normalizer, host="normalizer"):
			with ASGIAR(registry, host="registry"):
				async with httpx.AsyncClient() as client:
					response = await client.get("http://kp/test")
				assert response.status_code == 200
```

The solution we chose was to encapsulate these contexts within decorators. These decorators can be added to tests to provide functionality. We settled on five decorators to cover most of the functionality we needed:

- `with_kp_overlay`
- `with_registry_overlay`
- `with_norm_overlay`
- `with_response_overlay`
- `with_translator_overlay`

The first three are simple - they wrap the function with a context provider that calls the ASGIAR library with the specified external service. `with_response_overlay` allows specifying a static response. This is useful for testing what happens if a host is offline or returns a 500 error.

`with_translator_overlay` combines the normalizer, registry, and any number of KPs into a single decorator. There are many tests that require all of these services present, so having one utility to encapsulate this is helpful. Putting it all together, here is a full example of one test with our framework:

```python
@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
            MONDO:0005148(( category biolink:DiseaseOrPhenotypicFeature ))
            MONDO:0005148(( category biolink:Disease ))
        """
    }
)
async def test_duplicate_results():
    """
    Some KPs will advertise multiple operations from the biolink hierarchy.

    Test that we filter out duplicate results if we
    contact the KP multiple times.
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0(( category biolink:ChemicalSubstance ))
        n1(( category biolink:DiseaseOrPhenotypicFeature ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)
    assert_no_warnings_trapi(output)

    assert len(output['message']['results']) == 1
```

## Conclusion & Future Improvements

Overall, we are happy with the current version of the testing framework. As of writing, there are 35 tests which provide about 95% coverage of the code currently in use. The tests are easy to maintain and can cover nearly all responses from external services.

Code is never fully complete, and this testing framework is no exception. There are a number of improvements that we are looking to work on in the future including:

- Comprehensive output validation
- More standardization of external components
- Query and response visualization tools for faster debugging
