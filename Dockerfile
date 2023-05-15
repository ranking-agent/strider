# Use RENCI python base image
FROM renciorg/renci-python-image:v0.0.1

# Add image info
LABEL org.opencontainers.image.source https://github.com/ranking-agent/strider

ENV PYTHONHASHSEED=0

# set up requirements
WORKDIR /app

# make sure all is writeable for the nru USER later on
RUN chmod -R 777 .

# Install requirements
ADD requirements-lock.txt .
RUN pip install -r requirements-lock.txt

# switch to the non-root user (nru). defined in the base image
USER nru

# Copy in files
ADD . .

# Set up base for command and any variables
# that shouldn't be modified
ENTRYPOINT ["gunicorn", "strider.server:APP", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "0"]

# Variables that can be overriden
CMD [ "--bind", "0.0.0.0:5781", "--workers", "4", "--threads", "3"]
