FROM gcr.io/kaggle-gpu-images/python@sha256:36a3a467ee49c0ca0e8b3271288c67ca1fdc04cc10e4ddb96ce142858db3303a
MAINTAINER "Qovaxx"
ARG DEBIAN_FRONTEND


# Install nvidia apex
RUN git clone https://github.com/NVIDIA/apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex && rm -rf ./apex

# Install other system libraries
# cmake: Makefile
# libvips: pyvips
RUN apt-get update && \
    apt-get install -y \
        cmake \
        nano \
        htop \
        libvips && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache-dir --upgrade pip
RUN mkdir /server
COPY ./requirements/requirements.txt /server/requirements.txt
RUN pip install --no-cache-dir -r /server/requirements.txt

ARG PROJECT_DIRPATH
ENV PYTHONPATH "${PYTHONPATH}:${PROJECT_DIRPATH}"
WORKDIR ${PROJECT_DIRPATH}

# SSH settings
ARG DOCKER_USER_NAME
ARG DOCKER_USER_PASS
ARG DOCKER_SSH_PORT
RUN echo "${DOCKER_USER_NAME}:${DOCKER_USER_PASS}" | chpasswd
RUN apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo "${DOCKER_USER_NAME}:${DOCKER_USER_PASS}" | chpasswd && \
    sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed "s@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g" -i /etc/pam.d/sshd
EXPOSE ${DOCKER_SSH_PORT}
ENTRYPOINT /usr/sbin/sshd -D