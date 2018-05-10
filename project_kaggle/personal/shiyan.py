#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
import time

def do_one():
    index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
    df = pd.DataFrame({
    'http_status': [200, 200, 404, 404, 301],

    'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},

    index = index)
    print df
    new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10', 'Chrome']
    dd = df.reindex(new_index)
    print dd
def two():
    A = np.array([[1,1],[2,2]])
    B = np.array([[3,3],[4,4]])
    C = A + B
    print C
    print C/2

two()