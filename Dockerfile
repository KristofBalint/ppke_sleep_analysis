FROM python:3.10.1

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install system-level build tools to prevent 'greenlet/gevent' errors
#RUN apt-get update && apt-get install -y \
#    build-essential \
#    python3-dev \
#    libffi-dev \
#    && rm -rf /var/lib/apt/lists/*

## Upgrade pip and install requirements
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN git clone https://github.com/zou-group/sleepfm-clinical.git
