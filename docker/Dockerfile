
# Select a base image:

FROM python:3.7

#tensorflow/tensorflow:latest

#Use CUDA 10 (GPU drivers >= 410.48)
#tensorflow/tensorflow:1.13.1-gpu-py3

#Use CUDA 9 (GPU drivers >= 384.81)
#tensorflow/tensorflow:1.9.0-gpu

#Use CUDA 8 (GPU drivers >= 367.48)
#tensorflow/tensorflow:1.4.0-gpu

#nvidia/cuda:8.0-cudnn5-devel

#python:2.7
#python:3.7


# Install OS packages (from requirements_os.txt):

ADD docker/requirements_os.txt .
RUN apt update -y && apt install -y $(cat requirements_os.txt|grep -v '#')


# Install Python packages (from requirements.txt):

ADD docker/requirements.txt .
RUN pip install -r requirements.txt