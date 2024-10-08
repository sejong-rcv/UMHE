FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
LABEL maintainer "JM-Shin <jmshin@rcv.sejong.ac.kr>"

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG C.UTF-8

ARG PYTHON_VERSION=
ARG CONDA_ENV_NAME=

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
    apt-utils \
    build-essential \
    sudo \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ssh \
    wget \
    unzip \
    tmux
RUN rm -rf /var/lib/apt/lists/*

RUN curl -o /tmp/miniconda.sh -sSL http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

RUN conda update -y conda
RUN conda create -n $CONDA_ENV_NAME python=$PYTHON_VERSION

ENV PATH /usr/local/envs/$CONDA_ENV_NAME/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

COPY requirements.txt /tmp/requirements.txt

RUN source activate ${CONDA_ENV_NAME} && pip install --no-cache-dir -r /tmp/requirements.txt

RUN source activate ${CONDA_ENV_NAME} && conda install av -c conda-forge

# Pytorch
RUN source activate ${CONDA_ENV_NAME} && pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
