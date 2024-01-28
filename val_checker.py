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
print(X.head())
x_train = X.drop('RainTomorrow', axis=1)
y = X.RainTomorrow

loaded_model = pickle.load(open('models/LogisticRegression.pickle', 'rb'))
res = loaded_model.score(X, y)
print('Accuracy')
print(res)

