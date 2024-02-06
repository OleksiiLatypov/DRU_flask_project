import os
import pickle
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


from utils.dataloader import DataLoader
from settings.constants import TRAIN_CSV, SPECIFICATIONS, SAVED_ESTIMATOR, MODELS_FOLDER


with open(SPECIFICATIONS) as f:
    specifications = json.load(f)

raw_train = pd.read_csv(TRAIN_CSV)
x_columns = specifications['description']['X']
y_column = specifications['description']['y']

X_raw = raw_train[x_columns]

loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()
y = raw_train.RainTomorrow

model = GradientBoostingClassifier()
model.fit(X, y)

with open(os.path.join(MODELS_FOLDER, 'GradientBoostingClassifier.pickle'), 'wb') as f:
    pickle.dump(model, f)

if __name__ == '__main__':
    print('Model saved successfully !')