#!/usr/bin/env bash

export $(egrep -v '^#' .env | xargs)

# run workers and api server
supervisord -c ./supervisord.conf
