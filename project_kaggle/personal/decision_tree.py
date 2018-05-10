#coding=utf-8
from sklearn import tree
from model_two import trans_learndata
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import pandas as pd
from model_netron import *
from sklearn.ensemble import RandomForestClassifier

def get_treemodel(traindata,features):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(traindata[features],traindata['crime'])
    return clf
def get_randommodel(traindata,features):
    clf = RandomForestClassifier(n_estimators=100,max_depth=20)
    clf.fit(traindata[features],traindata['crime'])
    return clf


def get_treeresult():
    data_one, features = data_dealT('training.csv')
    data_two,features = data_dealT("validation.csv")
    model = get_randommodel(data_one, features)
    log = get_log(model, data_two, features)



def get_treetestresult():
    data_one,features,lecrime,crime = data_deal('train_new.csv')
    data_two = data_dealtest("test.csv")
    model = get_randommodel(data_one,features)
    log = get_test_log(model,data_two,features)
    categorys = get_category(lecrime, crime)
    final_re = pd.DataFrame(data=log, columns=categorys)
    final_re.to_csv('randomfour_model.csv')


def get_treetest():
    data_one,features,lecrime,crime = data_deal('train_new.csv')
    data_two = data_dealtest("test.csv")
    model = get_randommodel(data_one,features)
    log = get_test_log(model,data_two,features)
    return log


