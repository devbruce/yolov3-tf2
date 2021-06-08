FROM tensorflow/tensorflow:2.5.0-gpu


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

COPY preview /YOLOv3-TF2/preview/
COPY datasets /YOLOv3-TF2/datasets/
COPY ckpts /YOLOv3-TF2/ckpts/
COPY tutorial.ipynb /YOLOv3-TF2/
COPY README.md /YOLOv3-TF2/
COPY eval_coco.py /YOLOv3-TF2/
COPY train.py /YOLOv3-TF2/
COPY configs /YOLOv3-TF2/configs/
COPY libs /YOLOv3-TF2/libs/

WORKDIR /YOLOv3-TF2
