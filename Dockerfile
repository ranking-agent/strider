FROM python:3.9

# Add image info
LABEL org.opencontainers.image.source https://github.com/ranking-agent/strider

# set up requirements
WORKDIR /app

# Install requirements
ADD requirements-lock.txt .
RUN pip install -r requirements-lock.txt

# Copy in files
ADD . .

# set up base for command
ENTRYPOINT ["gunicorn", "strider.server:APP"]

# default variables that can be overriden
CMD [ "--bind", "0.0.0.0:5781" , "-k", "uvicorn.workers.UvicornWorker", "--workers", "17", "--threads", "3", "--worker-connections", "1000"]
