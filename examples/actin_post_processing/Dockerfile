FROM continuumio/miniconda3

RUN mkdir working

COPY env.yml .
RUN conda env create -f env.yml
SHELL ["conda", "run", "-n", "models", "/bin/bash", "-c"]

COPY src/ .
COPY post_processing.sh /post_processing.sh
RUN chmod +x /post_processing.sh

ENTRYPOINT /post_processing.sh