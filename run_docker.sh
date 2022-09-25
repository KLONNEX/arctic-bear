#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "Usage: bash run_docker.sh [IMAGE_NAME] [CONTAINER_NAME]"
    exit 1;
fi


docker run -it \
  --name "$CONTAINER_NAME" \
  -p 7777:7777 \
  --runtime=nvidia \
  --privileged=true \
  "$IMAGE_NAME" \
  /bin/bash
