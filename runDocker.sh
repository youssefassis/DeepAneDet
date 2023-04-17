#!/bin/bash
docker run -d --ipc=host --gpus all --user $(id -u):$(id -g) -it --rm  -v /home/user_name:/home/user_name -p 5000:8888 docker-name