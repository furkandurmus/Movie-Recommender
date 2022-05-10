FROM nvidia/cuda:11.3.0-base-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"


RUN apt update && apt install -y \
    wget 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda create -y -n recom_system python=3.8

COPY . src/ 

RUN /bin/bash -c "cd src \
    && source activate recom_system \
    && pip install -r requirements.txt"


