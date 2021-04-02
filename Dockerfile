FROM python:3.9

# Add image info
LABEL org.opencontainers.image.source https://github.com/ranking-agent/strider

# Install pipenv
RUN pip install pipenv

# Create working directory
WORKDIR /app

# Install dependencies
ADD Pipfile* .
RUN pipenv install --system

# Copy in files
ADD . .

# set up base for command
ENTRYPOINT ["uvicorn", "strider.server:APP"]

# default variables that can be overriden
CMD [ "--host", "0.0.0.0", "--port", "5781" ]
