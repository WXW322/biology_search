import pandas as pd


def get_data():
    data = pd.read_csv('B.csv')
    i = 1
    features = []
    while(i <= 420):
        features.append(str(i))
        i = i + 1
    data.columns = features
    data['lable'] = 0
    data.to_csv('B_new.csv')
    print (data.head())

get_data()

