import sys
import os
from os.path import join

# Add submodule path into import paths
PROJECT_FOLDER = os.path.dirname(__file__)
print('Project folder = {}'.format(PROJECT_FOLDER))
sys.path.append(join(PROJECT_FOLDER))
# Define the dataset folder and model folder based on environment
HOME_DATA_FOLDER = join(PROJECT_FOLDER, 'data')
os.makedirs(HOME_DATA_FOLDER, exist_ok=True)
print('*' * 35, ' path infor ', '*' * 35)
print('Data folder = {}'.format(HOME_DATA_FOLDER))
PRETRAINED_MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models')
os.makedirs(PRETRAINED_MODEL_FOLDER, exist_ok=True)
OUTPUT_FOLDER = join(HOME_DATA_FOLDER, 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
PREPROCESS_FOLDER = join(HOME_DATA_FOLDER, 'process_data')
os.makedirs(PREPROCESS_FOLDER, exist_ok=True)
print('Pretrained folder = {}'.format(HOME_DATA_FOLDER))