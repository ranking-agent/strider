#!/usr/bin/env bash

export $(egrep -v '^#' .env | xargs)

# run api server
uvicorn strider.server:APP --host 0.0.0.0 --port 5781
