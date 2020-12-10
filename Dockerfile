FROM python:3.9

# Add image info
LABEL org.opencontainers.image.source https://github.com/ranking-agent/strider

# install basic tools
RUN apt-get update
RUN apt-get install -yq \
    vim sudo

# set up murphy
ARG UID=1000
ARG GID=1000
RUN groupadd -o -g $GID murphy
RUN useradd -m -u $UID -g $GID -s /bin/bash murphy

# set up requirements
WORKDIR /home/murphy
ADD --chown=murphy:murphy ./requirements.txt .
RUN pip install -r /home/murphy/requirements.txt

# Copy in files
ADD --chown=murphy:murphy . .

# become murphy
ENV HOME=/home/murphy
ENV USER=murphy
USER murphy

# set up base for command
ENTRYPOINT ["uvicorn", "strider.server:APP"]

# default variables that can be overriden
CMD [ "--host", "0.0.0.0", "--port", "5781" ]
