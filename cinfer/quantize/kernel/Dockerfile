FROM nvcr.io/nvidia/pytorch:23.04-py3
 
RUN apt-get update \
    && apt-get install -y \
       software-properties-common \
       cmake \
       vim \
       sudo \
       git 
 
RUN apt-get install -y \
    openssh-server

RUN pip install transformers==4.31.0 bitsandbytes -i https://mirrors.cloud.tencent.com/pypi/simple
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation

WORKDIR /workspace
RUN git clone https://github.com/NetEase-FuXi/EETQ.git
WORKDIR /workspace/EETQ
RUN git submodule update --init --recursive
RUN pip install .