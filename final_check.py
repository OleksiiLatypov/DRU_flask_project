import json
from pprint import pprint

import requests
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import DataLoader, Estimator
from settings.constants import TRAIN_CSV, VAL_CSV

with open('settings/specifications.json') as f:
    specifications = json.load(f)

info = specifications['description']
x_columns, y_column, metrics = info['X'], info['y'], info['metrics']
pprint(x_columns)

train_set = pd.read_csv(TRAIN_CSV, header=0)
val_set = pd.read_csv(VAL_CSV, header=0)

train_x, train_y = train_set[x_columns], train_set[y_column]
print('TRAIN_Y\n', train_y)
val_x, val_y = val_set[x_columns], val_set[y_column]


loader = DataLoader()
loader.fit(val_x)
val_processed = loader.load_data()
print('data: ', val_processed[:10])

req_data = {'data': json.dumps(val_x.to_dict())}
response = requests.get('http://0.0.0.0:8000/predict', data=req_data)
api_predict = response.json()['prediction']
print('predict: ', api_predict[:10])

VAL_Y = val_processed['RainTomorrow']
print('Y:', VAL_Y[:10])
api_score = eval(metrics)(VAL_Y, api_predict)
print('accuracy: ', api_score)

if __name__ == '__main__':
    print()