#!/bin/bash

if [[ $# -ne 4 ]]; then
    echo "Usage: bash run_docker.sh [DATA_ROOT] [OUTPUT_ROOT] [IMAGE_NAME] [CONTAINER_NAME]"
    exit 1;
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

DATA_ROOT=$(get_real_path "$1")
OUTPUT_ROOT=$(get_real_path "$2")
IMAGE_NAME="$3"
CONTAINER_NAME="$4"

if [ !  -d "$DATA_ROOT" ]; then
  echo "Wrong path to the input dataset"
  exit 1;
fi

if [ !  -d "$OUTPUT_ROOT" ]; then
  mkdir "$OUTPUT_ROOT"
fi

docker run -it \
  --name "$CONTAINER_NAME" \
  -p 7777:7777 \
  -v "$DATA_ROOT:/server/dataset" \
  -v "$OUTPUT_ROOT:/server/outputs" \
  --runtime=nvidia \
  --privileged=true \
  "$IMAGE_NAME" \
  /bin/bash
