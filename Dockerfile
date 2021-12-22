FROM python:3.9.7

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
# CB Learn install https://github.com/dekuenstle/cblearn
RUN git clone https://github.com/dekuenstle/cblearn 
RUN cd cblearn && pip install .

COPY . .
CMD ["python", "experiment_runner.py", "experiments/02-small-clusters.yaml"]
