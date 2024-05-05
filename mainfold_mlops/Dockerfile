FROM python:3.8

RUN apt-get update && \
    apt-get install -y build-essential &&\
    apt-get install -y libpq-dev && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip

COPY . /code

RUN chmod +x /code/src

RUN pip install --no-cache-dir -r /code/src/requirements/requirements.txt


EXPOSE 8085

WORKDIR /code/src

# Set PYTHONPATH environment variable
ENV PYTHONPATH "${PYTHONPATH}:/code/src"

CMD pip install -e .

