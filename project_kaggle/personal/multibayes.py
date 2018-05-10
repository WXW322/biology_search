#coding=utf-8
from model_two import *
from sklearn.naive_bayes import MultinomialNB
from model_init import *
import datetime
from sklearn.externals import joblib

def data_deal(file_name):
    train = pd.read_csv(file_name, parse_dates=['Dates'])
    # 用LabelEncoder对不同的犯罪类型编号
    leCrime = preprocessing.LabelEncoder()
    crime = leCrime.fit_transform(train.Category)
    #cul_model = get_cluster(train['X'],train['Y'],1)
    traindata,features = trans_learndata(train)
    traindata['crime'] = crime
    #traindata['lo'] = cul_model.labels_
    features = features+['Address']
    #features = features + ['lo']
    dister = preprocessing.LabelEncoder()
    district = dister.fit_transform(train.Address)
    traindata['Address'] = district
    t_data = pd.concat([train['X'],train['Y']],axis=1)


    return traindata,features,leCrime,crime

def data_dealtest(file_name):
    test = pd.read_csv(file_name, parse_dates=['Dates'])
    # 用LabelEncoder对不同的犯罪类型编号
    #leCrime = preprocessing.LabelEncoder()
    #crime = leCrime.fit_transform(train.Category)
    traindata,features = trans_learndata(test)
    #traindata['crime'] = crime
    features = features+['Address']
    dister = preprocessing.LabelEncoder()
    district = dister.fit_transform(test.Address)
    traindata['Address'] = district
    t_data = pd.concat([test['X'], test['Y']], axis=1)
    return traindata,features

def get_multimodel(traindata,features):
    clf = MultinomialNB()
    clf.fit(traindata[features],traindata['crime'])
    strtime = datetime.datetime.now().strftime('%Y-%m-%d')
    joblib.dump(clf, 'multival_model/' + 'model' + strtime + '.pkl')
    return clf

def predict_multi(model,testData,features):
    predict = model.predict_proba(testData[features])
    print("多重贝叶斯log损失为 %f" % (log_loss(testData['crime'], predict)))
    return predict

def forecast(model,testData,features):
    predict = model.predict_proba(testData[features])
    return predict

def get_muldata():
    traindata,features,leCrime,crime = data_deal('train_new.csv')
    #testData,features = data_dealtest('test.csv')
    # #result = forecast(model,testData,features)
    # #categorys = get_category(leCrime,crime)
    # #final_re = pd.DataFrame(data=result,columns=categorys)
    strtime = datetime.datetime.now().strftime('%Y-%m-%d')
    #final_re.to_csv("multival_model/"+strtime+'.csv')
    #training, validation = train_test_split(traindata, test_size=0.6)
    validation,features,leCrime,crime = data_deal('validation.csv')
    model = get_multimodel(traindata,features)
    data = predict_multi(model,validation,features)
    return data,leCrime,crime

def getmultest():
    traindata, features, leCrime, crime = data_deal('train_new.csv')
    # testData,features = data_dealtest('test.csv')
    # #result = forecast(model,testData,features)
    # #categorys = get_category(leCrime,crime)
    # #final_re = pd.DataFrame(data=result,columns=categorys)
    strtime = datetime.datetime.now().strftime('%Y-%m-%d')
    # final_re.to_csv("multival_model/"+strtime+'.csv')
    # training, validation = train_test_split(traindata, test_size=0.6)
    validation, features = data_dealtest('test.csv')
    model = get_multimodel(traindata, features)
    data = forecast(model, validation, features)
    return data, leCrime, crime

