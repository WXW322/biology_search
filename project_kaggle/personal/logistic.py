#coding=utf-8
from model_two import *
from multibayes import *
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('train.csv', parse_dates=['Dates'])
traindata,features = trans_learndata(train)
leCrime = preprocessing.LabelEncoder()
crime = leCrime.fit_transform(train.Category)
traindata, features = trans_learndata(train)
traindata['crime'] = crime
features = features + ['Address']
dister = preprocessing.LabelEncoder()
district = dister.fit_transform(train.Address)
traindata['Address'] = district
training, validation = train_test_split(traindata, test_size=0.6)
model = LogisticRegression(C=0.1)
model.fit(training[features],training['crime'])
predicted = np.array(model.predict_proba(validation[features]))
print "逻辑回归log损失为 %f" %(log_loss(validation['crime'], predicted))