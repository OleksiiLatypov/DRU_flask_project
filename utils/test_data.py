import os.path
import pickle
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from utils.dataloader import DataLoader
from settings. constants import VAL_CSV, SPECIFICATIONS, SAVED_ESTIMATOR, MODELS_FOLDER


with open(SPECIFICATIONS) as f:
    specifications = json.load(f)

x_columns = specifications['description']['X']
y_column = specifications['description']['y']

raw_val = pd.read_csv(VAL_CSV)
x_raw = raw_val[x_columns]

loader = DataLoader()
loader.fit(x_raw)
X = loader.load_data()
y = raw_val.RainTomorrow

loaded_model = pickle.load(open(SAVED_ESTIMATOR, 'rb'))
acc = loaded_model.score(X, y)

if __name__ == '__main__':
    print('Accuracy:')
    print(acc)