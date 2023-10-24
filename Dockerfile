FROM swaggerapi/swagger-ui:v4.18.2 AS swagger-ui
FROM nvidia/cuda:11.7.1-base-ubuntu22.04

ENV PYTHON_VERSION=3.10

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

RUN python3 -m pip install -U pip setuptools

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN pip install uvicorn gunicorn fastapi python-multipart

CMD gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 0 app.api:app -k uvicorn.workers.UvicornWorker
