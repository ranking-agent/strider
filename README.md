
# Strider

__A web service and API for Strider, the knowledge-provider querying, answer generating, ranking module of ARAGORN.__

This service accepts a biomedical question as a [Translator reasoner standard message](https://github.com/NCATS-Tangerine/NCATS-ReasonerStdAPI) and asynchronously generates results in the same format.

## Demonstration

A live version of the API can be found [here](http://robokop.renci.org:5781/docs).

## Deployment

A docker file is included in the base directory and can be used to build the customized container

```bash
docker build -t strider .
```

Strider utilizes Neo4j as an internal graph cache, RabbitMQ for message passing, and Redis for local storage. These are provided and configured through standard container images and docker-compose. Strider requires a [custom Neo4j plugin](https://github.com/TranslatorIIPrototypes/strider-neo4j) to be placed in the neo4j_plugins directory. A pre-built version of the plugin can be download and saved to the correct location using

```bash
curl -L https://github.com/TranslatorIIPrototypes/strider-neo4j/releases/download/v1.0.0/strider-1.0.0.jar -o neo4j_plugins/strider-1.0.0.jar
```

A set of environmental variables must be defined in a `.env` file. Make sure to replace `***` with actual passwords.

```bash
SUPERVISOR_PORT=9000
SUPERVISOR_USER=murphy
SUPERVISOR_PASSWORD=***
RABBITMQ_USER=murphy
RABBITMQ_PASSWORD=***
KPREGISTRY_URL=http://robokop.renci.org:4983
OMNICORP_URL=http://robokop.renci.org:3210
BIOLINK_URL=https://bl-lookup-sri.renci.org
```

All necessary containers can then be built and started using docker-compose

```bash
docker-compose up
```

## Usage

`http://<HOST>:5781/docs`
