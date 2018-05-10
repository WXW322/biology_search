#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
import time
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

class narmol:
    def __init__(self,mean,var):
        self.mean = mean
        self.var = var
    def get_mean(self):
        return self.mean
    def get_var(self):
        return self.var

def trans_learndata(train):
    # 用LabelEncoder对不同的犯罪类型编号
    #leCrime = preprocessing.LabelEncoder()
    #crime = leCrime.fit_transform(train.Category)
    # get_lables(crime)
    # 因子化星期几，街区，小时等特征
    days = pd.get_dummies(train.DayOfWeek)
    district = pd.get_dummies(train.PdDistrict)
    hour = train.Dates.dt.hour
    #month = pd.get_dummies(train.Dates.dt.month)
    #month.columns = ['one','two','three','four','five','six','seven','eight','nine','ten','ele','twe']
    hour = pd.get_dummies(hour)
    # 组合特征
    trainData = pd.concat([ hour, days, district, train['X'], train['Y']], axis=1)
    #trainData = pd.concat([month,hour, days, district,train['X'],train['Y']], axis=1)
    #trainData['crime'] = crime
    features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
                'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
                'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
    hourFea = [x for x in range(0, 24)]
    features = features + hourFea
    #features = features + ['one','two','three','four','five','six','seven','eight','nine','ten','ele','twe']

    # 分割训练集(3/5)和测试集(2/5)
    # training, validation = train_test_split(trainData, test_size=0.6)
    return trainData,features

def KNN_get(data,lables):
    knn_model = KNeighborsClassifier(n_neighbors=100)
    knn_model.fit(data,lables)
    return knn_model
def knn_pre(data,model):
    result = model.predict(data)
    print result
    return result


def get_normal(train):
    category = train.Category.unique()
    t_X = {}
    t_Y = {}
    for ca in category:
        t_cax = train.loc[train['Category'] == ca]['X']
        t_cay = train.loc[train['Category'] == ca]['Y']
        t_norx = narmol(t_cax.mean(),t_cax.var())
        t_X[ca]=t_norx
        t_nory = narmol(t_cay.mean(),t_cay.var())
        t_Y[ca] = t_nory
    return t_X,t_Y

def train_data(training,features):
    model = BernoulliNB()
    nbStart = time.time()
    model.fit(training[features], training['crime'])
    return model

def get_proble(x,n_da):
    return (1.0/np.sqrt(2*np.pi*n_da.get_var()))*np.exp(-1*np.square(x-n_da.get_mean())/(2*n_da.get_var()))



def get_cluster(T_X,T_Y,N):
    t_data = pd.concat([T_X,T_Y],axis=1)
    clu_model = KMeans(n_clusters=N,random_state=0)
    clu_model.fit(t_data)
    return clu_model



