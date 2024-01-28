from sklearn.linear_model import LogisticRegression


import pickle
import json
import pandas as pd

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

y = X.RainTomorrow


model = LogisticRegression()
model.fit(X, y)
res = model.score(X, y)
with open('models/LogisticRegression.pickle', 'wb') as f:
    pickle.dump(model, f)

if __name__ == '__main__':
    print(res)
