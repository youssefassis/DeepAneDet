FROM nvidia/cuda:11.6.0-base-ubuntu20.04
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV USER_NAME user_name
ENV GROUP group_name
ENV HOME /home/${USER_NAME}
ENV PYTHONPATH="${PYTHONPATH}:/path/to/dataStrategy:/path/to/models:/path/to/ressources"

WORKDIR ${HOME}

COPY ./requirements.txt requirements.txt
RUN DEBIAN_FRONTEND=noninteractive apt-get update --fix-missing && \
    apt-get install -y git libglib2.0-0 libsm6 libxrender1 libxext6 build-essential && \
    apt-get clean                 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install -r requirements.txt  && \
    pip install hydra-core --upgrade --pre && \
    addgroup --gid 201559 ${GROUP}  && \
    adduser --uid 665802 --ingroup ${GROUP} --home ${HOME} --disabled-password --shell /bin/bash --gecos "" ${USER_NAME}  && \
    adduser ${USER_NAME} sudo && \
    jupyter notebook --generate-config
   
COPY jupyter_notebook_config.json ${HOME}/.jupyter/jupyter_notebook_config.json
RUN chown -R ${USER_NAME}:${GROUP} ${HOME}/.jupyter
USER ${USER_NAME}

EXPOSE 5000
VOLUME ${HOME}
CMD ["jupyter", "lab"]