FROM  jupyter/scipy-notebook:python-3.10.9

LABEL maintainer="TimeWz667 <timewz667@gmail.com>"
LABEL name="pkg/pkg"

COPY /requirements.txt .
RUN mkdir -p /home/jovyan/src
COPY src/ /home/jovyan/src

RUN pip install setuptools>=65.5.1
RUN pip install -r requirements.txt
RUN pip install -e /home/jovyan/src/

WORKDIR /home/jovyan/work

SHELL ["/bin/bash", "-o", "pipefail", "-c"]