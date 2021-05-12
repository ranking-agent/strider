FROM python:3.9

# Add image info
LABEL org.opencontainers.image.source https://github.com/ranking-agent/strider

# set up requirements
WORKDIR /app

# Install requirements
ADD requirements.txt .
RUN pip install -r requirements.txt

# set up base for command
ENTRYPOINT ["uvicorn", "strider.server:APP"]

# Copy in files
ADD . .

# default variables that can be overriden
CMD [ "--host", "0.0.0.0", "--port", "5781" ]
