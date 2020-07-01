#!/usr/bin/env bash
# Script for working with the main docker-compose commands.
# When building an utils, git attributes are written to environment variables

# Example usage:
#  base:
#   ./docker-compose.sh build
#   ./docker-compose.sh up server_name service_name
#   ./docker-compose.sh down server_name
#  extended options:
#   ./docker-compose.sh build --parallel --force-rm
#   ./docker-compose.sh up server_name service_name --no-color --build
#   ./docker-compose.sh down server_name -v --rmi

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

COMMAND=$1
SERVER=$2
PARAMS="${@:3}"
case ${COMMAND} in
    build)
        /home/local/X5/mikhail.chernyshev/docker-compose -f docker-compose.yml build ${PARAMS}
        ;;
    up)
        /home/local/X5/mikhail.chernyshev/docker-compose -f docker-compose.yml -f docker-compose.${SERVER,,}.yml up -d ${PARAMS}
        ;;
    down)
        /home/local/X5/mikhail.chernyshev/docker-compose -f docker-compose.yml -f docker-compose.${SERVER,,}.yml down ${PARAMS}
        ;;
    *)
        echo $"Usage: $0 {build|up|down}"
        exit 1
esac
