FROM ubuntu

RUN apt-get install -y \
    unzip

FROM python:3.9

WORKDIR /project

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN /bin/bash -c "echo pwd"