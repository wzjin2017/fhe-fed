FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]
RUN sed -i ~/.profile -e 's/mesg n || true/tty -s \&\& mesg n/g'
RUN apt update -y && apt upgrade -y
RUN apt-get update && apt-get install build-essential -y
RUN apt-get install git -y
RUN apt-get install cmake -y
RUN apt-get install autoconf -y
RUN apt-get install -y clang libomp5 libomp-dev
RUN apt-get install -y ninja-build libmpfr-dev libgmp-dev libboost-all-dev
RUN apt install vim -y
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.9 python3-distutils python3-pip python3-apt
RUN pip3 install pybind11

WORKDIR /root/sfl
RUN git clone https://github.com/weidai11/cryptopp.git
RUN cd cryptopp && make && make test && make install

WORKDIR /root/sfl
#RUN git clone -b release-v1.11.2 https://gitlab.com/palisade/palisade-development.git
# RUN mkdir -p /root/sfl/palisade-development/build
# WORKDIR /root/sfl/palisade-development/build
# RUN cmake .. && make && make install
RUN git clone https://gitlab.com/palisade/palisade-release.git
RUN mkdir -p /root/sfl/palisade-release/build
WORKDIR /root/sfl/palisade-release/build
RUN cmake .. && make && make install


COPY . /root/sfl/
WORKDIR /root/sfl/palisade_pybind/SHELFI_FHE/src
RUN pip3 install ../
WORKDIR /root/sfl

RUN pip3 install numpy

RUN pip3 install matplotlib pandas pytorch_lightning torch torchvision
RUN pip3 install -U scikit-learn
RUN pip3 install jupyterlab
