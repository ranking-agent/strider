FROM python:3.8.1-buster

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
ADD --chown=murphy:murphy ./requirements.txt /home/murphy/requirements.txt
RUN pip install -r /home/murphy/requirements.txt

# set up strider
ADD --chown=murphy:murphy ./strider /home/murphy/strider
ADD --chown=murphy:murphy ./setup.py /home/murphy/setup.py
RUN pip install -e .

# get meta-stuff
ADD --chown=murphy:murphy ./logging_setup.yml /home/murphy/logging_setup.yml
ADD --chown=murphy:murphy ./run_workers.py /home/murphy/run_workers.py
ADD --chown=murphy:murphy ./main.sh /home/murphy/main.sh

# become murphy
ENV HOME=/home/murphy
ENV USER=murphy
USER murphy

# set up entrypoint
CMD ["/home/murphy/main.sh"]