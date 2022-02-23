FROM 10.0-base-ubuntu14.04

RUN apt-get update
RUN apt-get install -y python3.5

COPY . .

RUN pip install -r requirements.txt