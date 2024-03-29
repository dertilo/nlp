FROM ufoym/deepo:pytorch-py36-cu100

RUN apt-get update && \
    apt-get install -y openssh-server && \
    apt-get -y install rsync

RUN pip install -U --no-cache-dir \
            scikit-learn==0.20.3 \
            gunicorn \
            certifi==2018.1.18 \
            chardet==3.0.4 \
            falcon==1.4.1 \
            idna==2.6 \
            python-mimeparse==1.6.0 \
            requests==2.18.4 \
            six==1.11.0 \
            urllib3==1.22 \
            bs4 \
            flair \
            spacy==2.1.4 \
            sklearn-crfsuite==0.3.6 \
            git+git://github.com/hyperopt/hyperopt.git \
            ray \
            psutil

RUN python -m spacy download en_core_web_sm


RUN pip install -U --no-cache-dir \
            git+https://git@gitlab.tubit.tu-berlin.de/tilo-himmelsbach/util.git

RUN git clone https://github.com/facebookresearch/ParlAI.git && \
    cd ./ParlAI && python setup.py develop

WORKDIR /docker-share
ENV PYTHONPATH /docker-share
