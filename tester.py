import pickle
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.dataloader import DataLoader
from settings.constants import VAL_CSV

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

y = raw_val.RainTomorrow
print(y.head())

loaded_model = pickle.load(open('models/LogisticRegression.pickle', 'rb'))
res = loaded_model.score(X, y)

if __name__ == '__main__':
    print('Accuracy')
    print(res)
