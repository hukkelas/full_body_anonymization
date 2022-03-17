FROM nvcr.io/nvidia/pytorch:21.12-py3
ARG UID=1000
ARG UNAME=testuser
ARG WANDB_API_KEY
RUN useradd -ms /bin/bash -u $UID $UNAME && \
    mkdir -p /home/${UNAME} &&\
    chown -R $UID /home/${UNAME}
WORKDIR /home/${UNAME}
ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY setup.py setup.py
RUN pip install -e .
RUN pip install wandb


ENV WANDB_API_KEY=$WANDB_API_KEY
WORKDIR /home/${UNAME}/
RUN pip install git+https://github.com/rwightman/pytorch-image-models

