#!/bin/bash
# JupyterLab on http://localhost:5000, with your home directory mounted at the same path
docker run -d --rm --name deepanedet --ipc=host --gpus all --user "$(id -u):$(id -g)" -v "$HOME:$HOME" -p 5000:8888 deepanedet
