FROM continuumio/miniconda:4.6.14

WORKDIR amazon
COPY . .

RUN apt-get update && apt-get install -y \
    unzip 

RUN cd /root/ && mkdir .kaggle && cd /amazon
RUN cp kaggle.json /root/.kaggle/kaggle.json
RUN conda env create -f yaml/amazon-env-dependencies.yaml

ARG init_mode=evaluate
ENV mode=$init_mode
ENTRYPOINT ["/bin/bash","-c","source activate amazon-access-env && bash main.sh $mode"]