import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

DATA_FOLDER = 'data'
MODELS_FOLDER = os.path.join(ROOT_DIR, 'models')
SETTINGS = 'settings'

TRAIN_CSV = os.path.join(ROOT_DIR, DATA_FOLDER, 'train.csv')
VAL_CSV = os.path.join(ROOT_DIR, DATA_FOLDER, 'val.csv')
SPECIFICATIONS = os.path.join(ROOT_DIR, SETTINGS, 'specifications.json')
SAVED_ESTIMATOR = os.path.join(MODELS_FOLDER, 'GradientBoostingClassifier.pickle')
print(MODELS_FOLDER)