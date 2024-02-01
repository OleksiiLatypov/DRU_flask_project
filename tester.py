import pickle
import json
import pandas as pd
from sklearn.svm import SVC

from utils.dataloader import DataLoader
from settings. constants import VAL_CSV


with open('settings/specifications.json') as f:
    specifications = json.load(f)

x_columns = specifications['description']['X']
y_column = specifications['description']['y']

raw_val = pd.read_csv(VAL_CSV)
x_raw = raw_val[x_columns]

loader = DataLoader()
loader.fit(x_raw)
X = loader.load_data()
x_test = X.drop('RainTomorrow', axis=1)
print(x_test.head())
y = X.RainTomorrow
print(y.head())

loaded_model = pickle.load(open('models/LogisticRegression.pickle', 'rb'))
r = loaded_model.score(x_test, y)
print(r)
print('success')