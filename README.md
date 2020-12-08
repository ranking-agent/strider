
# Strider

__A web service and API for Strider, the knowledge-provider querying, answer generating, ranking module of ARAGORN.__

This service accepts a biomedical question as a [Translator reasoner standard message](https://github.com/NCATSTranslator/ReasonerAPI) and asynchronously generates results in the same format.

## Demonstration

A live version of the API can be found [here](http://robokop.renci.org:5781/docs).

## Local Development

Docker-compose can be used to quickly set up a locally running version of the app. Once you have installed docker-compose you can run:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

This will start the requisite containers as well as the strider container. Changes made locally will update the container while running. 

## Running tests

Tests can be run through docker for convenience using the provided `Dockerfile.test`:

```bash
docker build -t strider-testing -f Dockerfile.test . 
docker run strider-testing
```

## Deployment

A docker-compose file is included for easy deployment. To use, you also must set up a .env file to specify URLs for external services. Example:

```
KPREGISTRY_URL=http://robokop.renci.org:4983
OMNICORP_URL=http://robokop.renci.org:3210
BIOLINK_URL=https://bl-lookup-sri.renci.org
```

After creating this at the root of the repository you can run:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

## Usage

`http://<HOST>:5781/docs`
