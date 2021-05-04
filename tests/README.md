# Testing Strider

### Content

* [`test_compatability.py`](test_compatability.py):

  We test that the knowledge portal correctly maps id CURIEs to the KPs' preferred prefixes and maps the results back to Striders preferred prefixes.

* [`test_query_planner.py`](test_query_planner.py):

  We test that query plans are correctly generated given a provided query graph along with the available KP registry and node normalization service.

* [`test_trapi.py`](test_trapi.py):

  We test utilities for manipulating TRAPI messages, including

  * merging messages
  * filtering results based on the query graph

* [`test_util.py`](test_util.py):

  We test miscellaneous utilities.

* [`test_server.py`](test_server.py):

  We test all Strider components together and their ability to correctly generate results given a provided query graph along with the available KPs, KP registry, and node normalization service.

### Workflow

Tests are run automatically via GitHub Actions whenever a pull request review is requested.

### Implementation

Testing an ARA that depends on external KPs, KP registry, and node normalization service is tricky. See [how we test Strider](../docs/TESTING_INFRASTRUCTURE.md).
