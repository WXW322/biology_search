#coding=utf-8
from model_two import *
from sklearn.neural_network import MLPClassifier
import datetime
from sklearn.externals import joblib
from model_init import *
def data_deal(file_name):
    train = pd.read_csv(file_name, parse_dates=['Dates'])
    # 用LabelEncoder对不同的犯罪类型编号
    leCrime = preprocessing.LabelEncoder()
    crime = leCrime.fit_transform(train.Category)
    traindata,features = trans_learndata(train)
    traindata['crime'] = crime
    traindata['X'] = traindata['X'] - traindata['X'].mean()
    traindata['Y'] = traindata['Y'] - traindata['Y'].mean()
    features = features+['X','Y']
    #scaler = preprocessing.MinMaxScaler()
    #X_scaled = scaler.fit_transform(train['address'])
    traindata['address'] = (train['address'] - train['address'].min())/(train['address'].max() - train['address'].min())
    features = features + ['address']
    return traindata,features,leCrime,crime

def data_dealT(file_name):
    train = pd.read_csv(file_name, parse_dates=['Dates'])
    # 用LabelEncoder对不同的犯罪类型编号
    leCrime = preprocessing.LabelEncoder()
    crime = leCrime.fit_transform(train.Category)
    traindata,features = trans_learndata(train)
    traindata['crime'] = crime
    traindata['X'] = traindata['X'] - traindata['X'].mean()
    traindata['Y'] = traindata['Y'] - traindata['Y'].mean()
    features = features+['X','Y']
    features = features + ['Address']
    # features = features + ['lo']
    dister = preprocessing.LabelEncoder()
    district = dister.fit_transform(train.Address)
    traindata['Address'] = district
    return traindata,features,leCrime,crime

def data_testT(file_name):
    train = pd.read_csv(file_name, parse_dates=['Dates'])
    # 用LabelEncoder对不同的犯罪类型编号
    traindata,features = trans_learndata(train)

    traindata['X'] = traindata['X'] - traindata['X'].mean()
    traindata['Y'] = traindata['Y'] - traindata['Y'].mean()
    features = features+['X','Y']
    features = features + ['Address']
    # features = features + ['lo']
    dister = preprocessing.LabelEncoder()
    district = dister.fit_transform(train.Address)
    traindata['Address'] = district
    return traindata,features

def data_dealtest(file_name):
    train = pd.read_csv(file_name, parse_dates=['Dates'])
    # 用LabelEncoder对不同的犯罪类型编号
   # leCrime = preprocessing.LabelEncoder()
   # crime = leCrime.fit_transform(train.Category)
    traindata, features = trans_learndata(train)
    #traindata['crime'] = crime
    traindata['X'] = traindata['X'] - traindata['X'].mean()
    traindata['Y'] = traindata['Y'] - traindata['Y'].mean()
    features = features + ['X', 'Y']
    traindata['address'] = (train['address'] - train['address'].min()) / (train['address'].max() - train['address'].min())
    features = features + ['address']
    return traindata


def get_netronmodel(trainData,features):
    clf = MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(40,40))
    clf.fit(trainData[features],trainData['crime'])
    strtime = datetime.datetime.now().strftime('%Y-%m-%d')
    #joblib.dump(clf,'neron_net/'+'model'+strtime+'.pkl')
    return clf

def get_log(model,testData,features):
    predict = model.predict_proba(testData[features])
    print("神经网络log损失为 %f" % (log_loss(testData['crime'], predict)))
    return predict

def get_test_log(model,testData,features):
    predict = model.predict_proba(testData[features])
    #print("神经网络log损失为 %f" % (log_loss(testData['crime'], predict)))
    return predict



def get_neronresult():
    traindata,features = data_deal('train_right.csv')
    training, validation = train_test_split(traindata, test_size=0.6)
    model = get_netronmodel(traindata,features)
    #validation,f = data_deal('validation.csv')
    #model = joblib.load("neron_net/model2017-12-22.pkl")
    data = get_log(model,validation,features)
    return data,validation

def get_nerontest():
    traindata, features,leCrime,crime = data_deal('train_right.csv')
    # training, validation = train_test_split(traindata, test_size=0.6)
    model = get_netronmodel(traindata, features)
    test = data_dealtest('test_right.csv')
    # model = joblib.load("neron_net/model2017-12-22.pkl")
    data = model.predict_proba(test[features])
    categorys = get_category(leCrime, crime)
    final_re = pd.DataFrame(data=data,columns=categorys)
    return final_re

def get_datatest_one():
    # 用LabelEncoder对不同的犯罪类型编号
    traindata, features, leCrime, crime = data_dealT('train_right.csv')
    model = get_netronmodel(traindata,features)
    test_data,features = data_testT('test_right.csv')
    data = model.predict_proba(test_data[features])
    catagory = get_category(leCrime,crime)
    final = pd.DataFrame(data = data,columns=catagory)
    final.to_csv('model_exp2.csv')

#get_datatest_one()









