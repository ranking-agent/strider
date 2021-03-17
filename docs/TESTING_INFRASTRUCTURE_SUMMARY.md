An ARA is a complex pieces of software. One of the most important tools for building complex software is testing. It's unclear how to implement effective testing for an ARA. The main challenge is that ARAs make calls to external tools that can behave (or misbehave) in a variety of ways. The Aragorn query-routing engine, known as Strider, contacts the following external tools:

- KP Registry to find KPs available to solve particular edges
- Node Normalizer to convert curies between formats
- Individual KPs to solve one-hop steps of a given query

During normal operation, these services could respond with 500 errors, JSON error codes, empty lists, or null values. One of the goals of the testing infrastructure is to verify that Strider gracefully handles these cases, even if that means submitting a log message and returning. To do this, we want to mock these external services. Mock services allow us to control the responses that these services send—opening up opportunities to test Strider in a controlled environment.

Our solution uses a feature of Python's httpcore library to intercept external HTTP calls and route them to internal handlers. All of this takes place within one Python process. This eliminates the need for networking infrastructure and makes the tests lightweight and easy to run locally.

The package for simulating external services is called ASGIAR - ASGI Augmented Reality. This allows overlaying an ASGI application, such as FastAPI or Django, to intercept HTTP requests. Any application written using any of these frameworks can "plug in" to ASGIAR and handle web requests.

 We use ASGIAR to attach mocked external services to Strider during testing, in particular the Node Normalizer, KP Registry, and KPs. These mock services live in separate repositories and can be installed as Python packages with pip. All of these services are built using FastAPI to maintain compatibility with ASGIAR. Each of them can be initialized with custom data during creation.


In addition to this infrastructure, our tests include helper functions that allow us to add these external service mocks using decorators. We also have helpers for specifying test data directly in the tests which makes tests very easy to understand. Here is an example of a finished test using all of our utilities:

```python
@pytest.mark.asyncio
@with_kp_overlay(
    name = "kp0",
    data = """
        CHEBI:6801(( category biolink:Drug ))
        MONDO:0005148(( category biolink:Disease ))
        CHEBI:6801-- predicate biolink:correlated_with -->MONDO:0005148
    """
)
@with_registry_overlay(
    """
    kp0 biolink:Drug -biolink:correlated_with-> biolink:Disease
    """
)
@with_normalizer_overlay()
async def test_reverse_lookup():
    """
    Test that we can look up an operation in reverse (subject -> object)
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( id MONDO:0005148 ))
        n0(( category biolink:Drug ))
        n1(( category biolink:Disease ))
        n0-- biolink:correlated_with -->n1
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

    # Validate output
    assert len(output['message']['results']) == 1
```

Overall, we are happy with the current version of the testing framework. As of writing, there are 35 tests which provide about 95% coverage of the code currently in use. The tests are easy to maintain and can cover nearly all responses from external services.
