FROM python:3.9-slim-buster


WORKDIR /dl-agent

ADD . .


RUN pip install -r /dl-agent/agent/requirements.txt

ENV AGENT_HOST 0.0.0.0
ENV AGENT_PORT 8888
ENV AGENT_VERBOSE 0

VOLUME /dl-agent/saved_models

EXPOSE $AGENT_PORT

CMD ["python", "agent/main.py"]
