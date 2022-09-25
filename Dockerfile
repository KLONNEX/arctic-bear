# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

RUN apt update && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /server

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install opencv-contrib-python==4.5.5.62

COPY . .
