FROM gcr.io/kaggle-gpu-images/python@sha256:36a3a467ee49c0ca0e8b3271288c67ca1fdc04cc10e4ddb96ce142858db3303a
MAINTAINER "Qovaxx"
ARG DEBIAN_FRONTEND

# TODO: remove auto competition and x5 directory
# Set proxy settings for x5 intranet
ARG PROXY
ENV HTTP_PROXY=http://${PROXY}
ENV HTTPS_PROXY=http://${PROXY}
ENV FTP_PROXY=http://${PROXY}
COPY ./x5/proxy.conf /etc/apt/apt.conf.d/proxy.conf

# TODO: remove auto competition and x5 directory
# Set ssl certificates for x5
COPY ./x5/x5.crt /usr/local/share/ca-certificates/x5.crt
COPY ./x5/x5_ca.crt /usr/local/share/ca-certificates/x5_ca.crt
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# TODO: remove auto competition and x5 directory
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    update-ca-certificates

# Install nvidia apex
RUN git clone https://github.com/NVIDIA/apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex && rm -rf ./apex

RUN pip install --no-cache-dir --upgrade pip
RUN mkdir /server
COPY ./requirements/requirements.txt /server/requirements.txt
#RUN pip uninstall -y lime
RUN pip install --no-cache-dir -r /server/requirements.txt

ARG PROJECT_DIRPATH
ENV PROJECT_DIRPATH ${PROJECT_DIRPATH}
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