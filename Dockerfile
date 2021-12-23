FROM python:3.9.7

WORKDIR /app

# General requirements
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# CB Learn install https://github.com/dekuenstle/cblearn
RUN git clone https://github.com/dekuenstle/cblearn 
RUN pip install ./cblearn

# Installs local tangles package
COPY tangles ./tangles
RUN pip install ./tangles

COPY experiments ./experiments
# Copy script files only now so we don't have to rebuild all the time
COPY *.py ./

CMD ["python", "experiment_runner.py", "experiments/02-small-clusters.yaml"]
