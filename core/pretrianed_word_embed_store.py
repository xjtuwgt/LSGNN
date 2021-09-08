import bcolz
import numpy as np
import io
import pickle #Parrallel precessing
from tqdm import tqdm
from time import time

def restore_fasttext_pretrained(pretrained_fast_text_file_name: str,
                                      dat_path_extend_name: str,
                                      word_pickle_name: str,
                                      word2index_name: str,
                                      customer_word_dict: dict=None):
    """
    :param pretrained_fast_text_file_name:
    :param dat_path_extend_name:
    :param word_pickle_name:
    :param word2index_name:
    :return:
    """
    words = []
    word2idx = {}
    idx = 0
    vectors = bcolz.carray(np.zeros(1), rootdir = f'{dat_path_extend_name}', mode = 'w')
    fin = io.open(pretrained_fast_text_file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_count, dim = map(int, fin.readline().split())
    print(word_count, dim)
    start_time = time()
    row_count = 0
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if customer_word_dict is not None:
            if word in customer_word_dict:
                vect = np.array(tokens[1:]).astype(np.float)
                words.append(word)
                word2idx[word] = idx
                idx = idx + 1
                vectors.append(vect)
        else:
            vect = np.array(tokens[1:]).astype(np.float)
            words.append(word)
            word2idx[word] = idx
            idx = idx + 1
            vectors.append(vect)
        row_count = row_count + 1

    word_count_ = idx
    vectors = bcolz.carray(vectors[1:].reshape(word_count_, dim), rootdir =
    f'{dat_path_extend_name}', mode = 'w')
    vectors.flush()
    pickle.dump(words, open(f"{word_pickle_name}", 'wb'))
    pickle.dump(word2idx, open(f"{word2index_name}", 'wb'))
    print('Processing {} {} takes {:.6f}'.format(row_count, len(words), time() - start_time))


def restore_glove_pretrained(dat_path_extend: str,
                             word_pickle_name: str,
                             word2index_name: str,
                          pretrained_glove_file_name: str,
                             dim=300,
                             customer_word_dict: dict=None):
    """
    :param glove_path:
    :param dat_path:
    :param pretrained_glove_file_name:
    :param word_count:
    :param dim:
    :return:
    """
    words = []
    word2idx = {}
    idx = 0
    start_time = time()
    vectors = bcolz.carray(np.zeros(1), rootdir = f'{dat_path_extend}', mode = 'w')
    row_count = 0
    with open(f'{pretrained_glove_file_name}', 'r') as f:
        for line in tqdm(f):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if customer_word_dict is not None:
                if word in customer_word_dict:
                    words.append(word)
                    word2idx[word] = idx
                    idx = idx + 1
                    vect = np.array(tokens[1:]).astype(np.float)
                    vectors.append(vect)
            else:
                words.append(word)
                word2idx[word] = idx
                idx = idx + 1
                vect = np.array(tokens[1:]).astype(np.float)
                vectors.append(vect)
            row_count = row_count + 1
    word_count = idx
    vectors = bcolz.carray(vectors[1:].reshape(word_count, dim), rootdir =
    f'{dat_path_extend}', mode = 'w')
    vectors.flush()
    pickle.dump(words, open(f"{word_pickle_name}", 'wb'))
    pickle.dump(word2idx, open(f"{word2index_name}", 'wb'))
    print('Processing {} {} takes {:.6f}'.format(row_count, len(words), time() - start_time))