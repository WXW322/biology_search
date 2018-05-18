import pandas as pd
from sklearn.neural_network import MLPClassifier

def train_model(epoch,rate,features):
    data_A = pd.read_csv('A_new.csv')
    data_B = pd.read_csv('B_new.csv')
    model = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(20,20,20))
    for i in range(0,epoch):
        data_a = data_A.sample(frac = rate)
        data_b = data_B.sample(frac = rate)
        data_f = pd.concat([data_a,data_b],axis = 0)
        #data_f = data_a.concat(data_b,axis = 0)
        model.fit(data_f[features],data_f['lable'])
    data_fone = data_A.sample(frac = rate)
    data_ftwo = data_B.sample(frac = rate)
    data_f = pd.concat([data_fone,data_ftwo],axis = 0)
    #data_f = data_fone.concat(data_ftwo,axis = 0)
    print (model.score(data_f[features],data_f['lable']))


features = []
i = 1
while(i <= 420):
    features.append(str(i))
    i = i + 1
train_model(40,0.1,features)
