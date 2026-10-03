FROM nvcr.io/nvidia/pytorch:20.12-py3
ARG env_det_num_threads=6
ARG env_det_verbose=1
ARG OMP_NUM_THREADS=1

ENV DEBIAN_FRONTEND noninteractive 
ENV NB_USER yassis 
ENV HOME /home/${NB_USER}
ENV det_num_threads $env_det_num_threads
ENV det_verbose $env_det_verbose 
ENV OMP_NUM_THREADS ${OMP_NUM_THREADS}

RUN apt-get update --fix-missing && apt-get install -y \
    git cmake make wget gnupg build-essential \
    software-properties-common gdb ninja-build

COPY ./requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt \
 && pip install hydra-core --upgrade --pre

COPY . .
RUN FORCE_CUDA=1 pip install -e .

RUN addgroup --gid 201559 ${NB_USER}
RUN adduser --uid 665802 --ingroup ${NB_USER} --home ${HOME} --disabled-password --shell /bin/bash --gecos "" ${NB_USER}
RUN adduser ${NB_USER} sudo

WORKDIR ${HOME}/NeuroDL

RUN jupyter notebook --generate-config
COPY jupyter_notebook_config.json ${HOME}/.jupyter/jupyter_notebook_config.json
RUN chown -R ${NB_USER}:${NB_USER} ${HOME}/.jupyter
USER ${NB_USER}

EXPOSE 5000
VOLUME ${HOME}

ENV PYTHONPATH "${PYTHONPATH}:/path/to/git/aneurysms/dataStrategy:/path/to/git/aneurysms/models:/path/to/git/aneurysms/ressources"
CMD ["jupyter", "lab"]  
