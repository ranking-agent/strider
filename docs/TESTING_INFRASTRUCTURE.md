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

## Context Providers

## Utilities

## Conclusion & Future Improvements

