
# Strider

__A web service and API for Strider, the knowledge-provider querying, answer generating module of ARAGORN.__

This service accepts a biomedical question as a [Translator reasoner standard message](https://github.com/NCATSTranslator/ReasonerAPI) and asynchronously generates results in the same format.

## Demonstration

A live version of the API can be found [here](https://strider.renci.org/docs).

## Local Development

### Management Script

The codebase comes with a zero-dependency python management script that can be used to automate basic local development tasks. Make sure you have docker and docker-compose installed and then run:

```bash
./manage.py dev # starts server accessible at 5781
./manage.py test # run tests
./manage.py coverage # run tests, open coverage report in browser
```

### Without Management Script

The management script is a simple alias for docker-compose commands. To run a dev environment without it you can use:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

This will start the requisite containers as well as the strider container. Changes made locally will update the container while running. 

You can also run tests and coverage reports without the management script. Check the `manage.py` file for instructions on how to do this.

### Profiler

The local development environment also includes a built-in profiler for debugging performance issues. To use this, set `PROFILER=true` in a `.env` file in the root of the repository. Once the application is running the profiler will automatically be run on all incoming requests. We haven't found a great asynchronous python profiler, but the current "best" one is pyinstrument. When the profiler is enabled, a browser page will open after a query has completed that shows the profile.

## Testing

Documentation for testing can be found in the [tests README](tests/README.md). Additional high level testing architecture overview can be found in the [docs folder](docs/TESTING_INFRASTRUCTURE.md). 

## Deployment

A docker-compose file is included for easy deployment. To use, you also must set up a .env file to specify some configuration options:

* `KPREGISTRY_URL` - The url for an independent KP registry server (https://github.com/ranking-agent/kp_registry).
* `OMNICORP_URL` - The url for an independent omnicorp server.
* `OPENAPI_SERVER_URL` - The url at which _this_ Strider instance will be accessible - for the purpose of generating a portable OpenAPI schema, e.g. for use with the [SmartAPI registry](https://smart-api.info/registry?q=strider).

example:
```
KPREGISTRY_URL=http://robokop.renci.org:4983
OMNICORP_URL=http://robokop.renci.org:3210
OPENAPI_SERVER_URL=https://strider.renci.org/
```

After creating this at the root of the repository you can run:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

## Usage

`http://<HOST>:5781/docs`

## Documentation

High level documentation can be found in the [docs folder](docs/README.md).
