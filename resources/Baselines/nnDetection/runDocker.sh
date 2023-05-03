#!/bin/bash
docker run -d --ipc=host --gpus all --user $(id -u):$(id -g) -it --rm  -v /path/to:/path/to -p 5000:8888 nndetection-docker
