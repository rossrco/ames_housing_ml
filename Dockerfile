FROM python:3.7.6-stretch

WORKDIR /usr/src/app

RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ADD /trainer /trainer
WORKDIR /trainer

ENTRYPOINT [ "python", "trainer.py" ]