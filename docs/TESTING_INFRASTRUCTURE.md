# Unit Testing Infrastructure Overview



## Introduction & Goals

An ARA is a complex pieces of software. One of the most important tools for building complex software is testing. It's less clear *how* to implement effective testing for an ARA. The main challenge is that ARAs make calls to external tools that can behave (or misbehave) in a variety of ways. When Strider receives a query it contacts the following external tools:

- Node Normalizer to convert curies between formats
- Individual KPs to solve one-hop steps of a given query

Comprehensive testing requires that we can control how these services respond. During normal operation, these services could respond with 500 errors, JSON error codes, empty lists, or null values. One of the goals of the testing infrastructure is to verify that Strider gracefully handles these cases, even if that means submitting a log message and returning.

A good example of this is how Strider handles node normalizer responses. One of the ways we use the node normalizer is to find synonyms. Using synonyms when contacting KPs causes them to return more results. Contacting the node normalizer is helpful, but not necessary. It is important for us to verify that Strider gracefully handles when the normalizer is unavailable. Specifically, we would like to ensure that a valid ressponse is returned, with results, and including a log message saying that the normalizer was unavailable. One of the high level goals of the testing framework is to support testing these types of use cases.

## Architecture Overview

A common testing pattern for large pieces of software is to build integration tests. This would involve running Strider, Node Normalizer, and individual KPs in separate processes. There are drawbacks to this approach that make it difficult. One main drawback is that testing requires a networking infrastructure. This means there is additional tooling that must be present to run tests.

Our infrastructure uses HTTPXMock to mock all external responses so the tests are never calling out to actual external services, and we can mock various response types to test how we handle errors.

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

## Conclusion & Future Improvements

Overall, we are happy with the current version of the testing framework. As of writing, there are 81 tests which provide about 72% coverage of the code currently in use. The tests are easy to maintain and can cover nearly all responses from external services.

Code is never fully complete, and this testing framework is no exception. There are a number of improvements that we are looking to work on in the future including:

- Comprehensive output validation
- More standardization of external components
- Query and response visualization tools for faster debugging
