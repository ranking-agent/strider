# Testing Strider

Each test file in `tests/` relates to a file in `strider/` and tries its best to test those functions specifically.

### Workflow

Tests are run automatically via GitHub Actions whenever a pull request review is requested.

### Implementation

Testing an ARA that depends on external KPs, KP registry, and node normalization service needs to be mocked so we don't actually call those external services.
See existing tests for examples on how to mock the services that your test needs.
