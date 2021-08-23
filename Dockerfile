FROM tensorflow/tensorflow:2.5.0-gpu

LABEL maintainer "devbruce"
LABEL email "bruce93k@gmail.com"
LABEL title "yolov3-tf2"

ARG DEBCONF_NOWARNINGS="yes"
ARG DEBIAN_FRONTEND="noninteractive"
ENV LC_ALL "C.UTF-8"
ENV PIP_NO_CACHE_DIR "1"
ENV PYTHONPATH "/yolov3-tf2:${PYTHONPATH}"


RUN apt-get update
RUN apt-get install -y vim

# Requirement of opencv-python
RUN apt-get install -y libgl1-mesa-glx

# Install python packages
RUN pip install --upgrade pip
RUN pip install opencv-python==4.5.2.52
RUN pip install matplotlib==3.3.4
RUN pip install albumentations==1.0.0
RUN pip install pycocotools==2.0.2
RUN pip install tqdm==4.61.0
RUN pip install jupyterlab==3.0.16

COPY preview /yolov3-tf2/preview/
COPY datasets /yolov3-tf2/datasets/
COPY ckpts /yolov3-tf2/ckpts/
COPY tutorial.ipynb /yolov3-tf2/
COPY README.md /yolov3-tf2/
COPY eval_coco.py /yolov3-tf2/
COPY train.py /yolov3-tf2/
COPY configs /yolov3-tf2/configs/
COPY libs /yolov3-tf2/libs/

WORKDIR /yolov3-tf2
