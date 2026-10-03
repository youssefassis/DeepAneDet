FROM python:3.12-slim

# PyTorch build: cu126 (NVIDIA GPU, run with --gpus all) or cpu
ARG TORCH=cu126
ENV PYTHONPATH=/app/dataStrategy:/app/models:/app/resources \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user

WORKDIR /app

# Dependencies first, so that code changes don't reinstall them
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/${TORCH} && \
    pip install -r requirements.txt

COPY jupyter_server_config.json /etc/jupyter/
COPY . .

# Writable home for any user id (runDocker.sh runs as the host user)
RUN mkdir -p ${HOME} && chmod 777 ${HOME}

EXPOSE 8888
CMD ["jupyter", "lab"]
