FROM bveeramani/openvino:r1.1

ENV DEVICE CPU

COPY . /app

WORKDIR ~
