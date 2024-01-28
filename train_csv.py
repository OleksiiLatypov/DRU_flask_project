from sklearn.linear_model import LogisticRegression

from settings.constants import TRAIN_CSV
from utils.dataloader import DataLoader
import pandas as pd

# train = pd.read_csv(TRAIN_CSV)
#
# #X_raw = train.drop("RainTomorrow", axis=1)
#
# #print(X_raw.head())
# print(100*'>')
#
# loader = DataLoader()
# loader.fit(train)
# X_raw = loader.load_data()
#
# print(100*'=')
# X = X_raw.drop('RainTomorrow', axis=1)
# print(X.head())
#
# print(100 * '==')
# y = X_raw["RainTomorrow"]
# print(y.head())
#
# print(X.head())

import pickle
import json
import pandas as pd
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

y = X.RainTomorrow


model = LogisticRegression()
model.fit(X, y)
res = model.score(X, y)
with open('models/LogisticRegression.pickle', 'wb') as f:
    pickle.dump(model, f)

if __name__ == '__main__':
    print(res)
