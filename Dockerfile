FROM python:3.7.6-stretch

WORKDIR /usr/src/app

RUN pip install --upgrade pip

ADD /trainer /trainer
WORKDIR /trainer

ENTRYPOINT [ "python", "trainer.py" ]