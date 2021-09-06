from tqdm import tqdm
import gzip, pickle
import io
import json
from time import time
import pandas as pd
from pandas import DataFrame

def load_fast_text_vectors(file_name):
    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    print(n, d)
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def load_json_file(file_name):
    start_time = time()
    with open(file_name, 'r') as fin:
        print('loading', file_name)
        data = json.load(fin)
    print('Loading {} from {} in {:.4f} seconds'.format(len(data), file_name, time() - start_time))
    return data

def load_gz_file(file_name):
    start_time = time()
    with gzip.open(file_name, 'rb') as fin:
        print('loading', file_name)
        data = pickle.load(fin)
    print('Loading {} from {} in {:.4f} seconds'.format(len(data), file_name, time() - start_time))
    return data

def save_dataframe2json(dataframe: DataFrame, filename: str, orient='records'):
    with open(filename, "w") as output_file:
        output_file.write(dataframe.to_json(orient=orient))
    print('saving {} into {}'.format(dataframe.shape, filename))

def save_dataframe2pickle(dataframe: DataFrame, filename: str):
    dataframe.to_pickle(filename)
    print('saving {} into {}'.format(dataframe.shape, filename))

def loadPickleAsDataFrame(pickle_file_name):
    dataframe = pd.read_pickle(pickle_file_name)
    return dataframe

def loadJSonAsDataFrame(jsonFileName):
    """
    :param jsonFileName:
    :return:
    """
    start = time()
    with open(jsonFileName, 'r') as f:
        json_data = json.load(f)
        dataFrame = pd.DataFrame(json_data)
    print("Json data loading takes {} seconds".format(time() - start))
    return dataFrame

def load_train_wikihop_data(train_data_name):
    """
    id, query. answer. candidates. supports
    :param train_data_name:
    :return:
    """
    data = load_json_file(file_name=train_data_name)
    return data

def load_dev_wikihop_data(dev_data_name):
    """
    id, query, answer, candidates. annotations, supports
    :param dev_data_name:
    :return:
    """
    data = load_json_file(file_name=dev_data_name)
    return data

