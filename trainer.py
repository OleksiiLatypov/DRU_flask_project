import pickle
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from utils.dataloader import DataLoader
from settings.constants import TRAIN_CSV


with open('settings/specifications.json') as f:
    specifications = json.load(f)

raw_train = pd.read_csv(TRAIN_CSV)
x_columns = specifications['description']['X']
y_column = specifications['description']['y']

X_raw = raw_train[x_columns]

loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()
print(X.head())
x_train = X.drop('RainTomorrow', axis=1)
print(x_train.head())
y = X.RainTomorrow
print(y.head())

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x_train, y)
with open('models/LogisticRegression.pickle', 'wb')as f:
    pickle.dump(model, f)

print('Success')