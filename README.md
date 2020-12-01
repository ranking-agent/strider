
# Strider

__A web service and API for Strider, the knowledge-provider querying, answer generating, ranking module of ARAGORN.__

This service accepts a biomedical question as a [Translator reasoner standard message](https://github.com/NCATSTranslator/ReasonerAPI) and asynchronously generates results in the same format.

## Demonstration

A live version of the API can be found [here](http://robokop.renci.org:5781/docs).

## Deployment

A docker file is included in the base directory and can be used to build the customized container

```bash
docker build -t strider .
```

A set of environmental variables must be defined in a `.env` file. Make sure to replace `***` with actual passwords.

```bash
OMNICORP_URL=http://robokop.renci.org:3210
BIOLINK_URL=https://bl-lookup-sri.renci.org
```

All necessary containers can then be built and started using docker-compose

```bash
docker-compose up
```

## Usage

`http://<HOST>:5781/docs`
