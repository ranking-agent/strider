
# Strider

__A web service and API for Strider, the knowledge-provider querying, answer generating, ranking module of ARAGORN.__

This service accepts a biomedical question as a [Translator reasoner standard message](https://github.com/NCATSTranslator/ReasonerAPI) and asynchronously generates results in the same format.

## Demonstration

A live version of the API can be found [here](http://robokop.renci.org:5781/docs).

## Local Development

### Management Script

The codebase comes with a zero-dependency python management script that can be used to automate basic local development tasks. Make sure you have docker and docker-compose installed and then run:

```bash
python3 manage.py dev # starts server accessible at 5781
python3 manage.py test # run tests
python3 manage.py coverage # run tests, open coverage report in browser
```

### Without Management Script

The management script is a simple alias for docker-compose commands. To run a dev environment without it you can use:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

This will start the requisite containers as well as the strider container. Changes made locally will update the container while running. 

You can also run tests and coverage reports withou the management script. Check the `manage.py` file for instructions on how to do this.

## Deployment

A docker-compose file is included for easy deployment. To use, you also must set up a .env file to specify URLs for external services. Example:

```
KPREGISTRY_URL=http://robokop.renci.org:4983
OMNICORP_URL=http://robokop.renci.org:3210
REDIS_URL=redis://localhost
SERVER_URL=https://robokop.renci.org:5781
```

After creating this at the root of the repository you can run:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

## Usage

`http://<HOST>:5781/docs`
