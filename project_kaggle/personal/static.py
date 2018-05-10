#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_two import *
from model_netron import *
from multibayes import *
from decision_tree import *

def draw_twopic(X,Y):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title('Scatter Plot')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    # 画散点图
    ax1.scatter(x=X, y=Y, c='r', marker='o')
    # 设置图标
    plt.legend('x1')
    # 显示所画的图
    plt.show()

def cut_data(train_data):
    print train_data.shape
    new_data = train_data.drop(train_data[train_data.Y >60].index)
    new_data.to_csv('train_new.csv')

def model_sum():
    train = pd.read_csv('train_new.csv',parse_dates=['Dates'])
    train_one,features = trans_learndata(train)
    leCrime = preprocessing.LabelEncoder()
    crime = leCrime.fit_transform(train.Category)
    train_one['crime'] = crime
    feayures_two = features+['X','Y']
    features = features+['Address']
    dister = preprocessing.LabelEncoder()
    district = dister.fit_transform(train.Address)
    train_one['Address'] = district
    feature_three = features + ['Address']
    training, validation = train_test_split(train_one, test_size=0.6)
    model_two = get_multimodel(training,feature_three)
    pro_two = model_two.predict_proba(validation[feature_three])
    print("神经网络log损失为 %f" % (log_loss(validation['crime'], pro_two)))

def traverse_data():
    train = pd.read_csv('train_new.csv', parse_dates=['Dates'])


def get_lables(T_L,N):
    length = T_L.max() - T_L.min()
    gap = length/N
    F_L = []
    for t_l in T_L:
        t_dis = t_l - T_L.min()
        t_lable = int(t_dis/gap)
        F_L.append(t_lable)
    return F_L

def traverse_data():
    train = pd.read_csv('test.csv')
    address = train['Address'].copy()
    result = train['Address'].unique()
    add_num = {}
    for r in result:
        num = train[train['Address'] == r].count()
        add_num[r] = num['Address']
        address[address==r] = add_num[r]
    train['address'] = address
    train.to_csv('test_right.csv')




def get_location():
    train = pd.read_csv('train_new.csv')
    leCrime = preprocessing.LabelEncoder()
    crime = leCrime.fit_transform(train.Category)
    train['crime'] = crime
    for i in range(0,32):
        t_da = train[train['crime'] == i]
        print ('class:', i)
        print t_da['X'].describe()
        print t_da['Y'].describe()

def splitdata():
    train = pd.read_csv('train_new.csv')
    trainning,validation = train_test_split(train,test_size=0.6)
    trainning.to_csv('training.csv')
    validation.to_csv('validation.csv')

def model_mix():
    data_one,validation = get_neronresult()
    data_two = get_muldata()
    data_f = 0.6*data_two + 0.4*data_one
    print("神经网络log损失为 %f" % (log_loss(validation['crime'], data_f)))

def model_mix_test():
    data_one = get_nerontest()
    data_two,lecrime,crime = getmultest()
    data_three = get_treetest()
    data_f = 0.6*data_two + 0.2*data_one + 0.2*data_three
    categorys = get_category(lecrime, crime)
    final_re = pd.DataFrame(data=data_f,columns=categorys)
    strtime = datetime.datetime.now().strftime('%Y-%m-%d')
    final_re.to_csv("mix_model" + strtime + '.csv')

def draw_kind():
    data = pd.read_csv('train.csv')

    da = data.groupby('breed').count().plot(kind = 'bar')
    da.figure
    plt.show()

def tt():
    dada = {'A':[1,1,3,3],'B':[1,2,3,4]}
    data = pd.DataFrame(data = dada)
    pl = data.groupby('A').count().plot(kind = 'bar')
    pl.figure
    plt.show()

draw_kind()
