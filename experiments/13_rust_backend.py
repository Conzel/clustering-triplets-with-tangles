"""
Small demo of the Rust backend. 
Currently shows subpar runtime and wrong results :(
"""
from questionnaire import Questionnaire
from experiment_runner import tangles_hard_predict
from sklearn.datasets import load_wine

x, y = load_wine(return_X_y=True)
q = Questionnaire.from_metric(x).values
y1 = tangles_hard_predict(q, 5, disable_rust_backend=False)
y2 = tangles_hard_predict(q, 5, disable_rust_backend=True)
