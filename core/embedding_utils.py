import numpy as np
from time import time
import bcolz
import torch
from torch import LongTensor
import math
import pickle #Parrallel precessing
from torch import nn

QUERY_START = '[Q]'
QUERY_END = '[/Q]'
ENTITY_START = '[E]'
ENTITY_END = '[/E]'
SEP_TOKEN = '[S]'
UNKNOWN = '[UNK]'
PAD_TOKEN = '[PAD]'
SPECIAL_TOKENS = (('query_start', QUERY_START), ('query_end', QUERY_END), ('entity_start', ENTITY_START),
                  ('entity_end', ENTITY_END), ('sep_token', SEP_TOKEN))
def load_pretrained_embedding_ndarray(embeding_file_name: str, dim=300,
                                     oov_default='zero', special_tokens=SPECIAL_TOKENS)->(np.ndarray, dict):
    """
    :param embeding_file_name:
    :param dim:
    :param oov_default:
    :return:
    """
    start_time = time()
    zeros_vector = np.zeros((1, dim), dtype=np.float32)#Also for padding, the last row
    if oov_default == 'zero':
        defaut_vector = zeros_vector
    else:
        defaut_vector = np.random.uniform(-0.1, 0.1, (1, dim))
    vectors = bcolz.open(f'{embeding_file_name}.dat')[:]
    word2idx = pickle.load(open(f'{embeding_file_name}.idx.pkl', 'rb'))
    vocab_size = len(word2idx)
    word2idx[UNKNOWN] = len(word2idx) + 1
    word2idx[PAD_TOKEN] = len(word2idx) + 1
    word_num = len(word2idx)
    for idx, special_word_pair in enumerate(special_tokens):
        word2idx[special_word_pair[1]] = word_num + idx
    special_token_vectors = np.random.uniform(-0.1, 0.1, (len(special_tokens), dim))
    special_token_dict = {_[0]: _[1] for _ in special_tokens}
    word2vec = np.vstack((vectors, defaut_vector, zeros_vector, special_token_vectors)).astype('float32', casting= 'same_kind')
    print('Loading word2vec {} from {} takes {:.6f} seconds'.format(vectors.shape, embeding_file_name, time() - start_time))
    assert word2vec.shape[0] == len(word2idx)
    return (word2vec, word2idx, vocab_size, special_token_dict)

class WordEmbedding(nn.Module):
    def __init__(self, pre_trained_name: str, oov_default='zero', dim=300, freeze=False):
        super(WordEmbedding, self).__init__()
        self.oov_default = oov_default
        self.dim = dim
        start_time = time()
        word2vec, self.word2idx, vocab_size, self.special_token_dict = \
            load_pretrained_embedding_ndarray(pre_trained_name, self.dim, self.oov_default)
        self.word2vec = torch.from_numpy(word2vec)
        self.wordEmbed = nn.Embedding.from_pretrained(embeddings=self.word2vec, freeze=freeze)
        self.word_count = len(self.word2idx)
        self.oov_idx = vocab_size
        self.pad_idx = vocab_size + 1
        print('Pre-trained embeddings loaded in {:.6f} seconds!'.format(time() - start_time))

    def forward(self, idxes: LongTensor):
        return self.wordEmbed(idxes)

    def decode_word2d_list(self, words_list):
        return [self.decode_words(words) for words in words_list]

    def decode_words(self, words):
        word_idxs = [self.word2idx.get(word, self.word_count) for word in words]
        return word_idxs

    def decode_word(self, word):
        idx = self.word2idx.get(word, self.oov_idx)
        return idx

    def padding_idx(self):
        return self.pad_idx

    def special_tokens(self):
        return self.special_token_dict

class RelationEmbedding(nn.Module):
    def __init__(self, num_relations, dim=300, gamma=0.1):
        super(RelationEmbedding, self).__init__()
        self.epsilon = 2.0
        self.gamma = gamma
        self.gain = (self.gamma + self.epsilon) / (5.0 * dim)
        self.relEmbbed = nn.Parameter(torch.zeros(num_relations, dim))
        nn.init.xavier_normal_(tensor=self.relEmbbed.data, gain=self.gain)

    def forward(self, idxes: LongTensor):
        return self.relEmbbed[idxes]

class PositionEmbedding(nn.Module):
    def __init__(self, max_position: int = 15000, hidden_dim=300, initial_type='sin_cos', freeze=False):
        super(PositionEmbedding, self).__init__()
        self.max_position = max_position
        self.hidden_dim = hidden_dim
        self.initial_type = initial_type
        self.freeze = freeze
        if self.initial_type == 'sin_cos':
            self.positionEmbed = nn.Embedding.from_pretrained(embeddings=self.sin_cos_position_initial(), freeze=self.freeze)
        elif self.initial_type == 'uniform':
            self.positionEmbed = nn.Embedding(max_position, hidden_dim)
            self.position_initial()
        else:
            print('Initial type: {} is not supported'.format(self.initial_type))
            exit(1)

    def sin_cos_position_initial(self):
        position = torch.arange(self.max_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000.0) / self.hidden_dim))
        pe = torch.zeros(self.max_position, 1, self.hidden_dim)
        gain = 2.0 / (self.hidden_dim + 4.0 * self.hidden_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term) * gain
        pe[:, 0, 1::2] = torch.cos(position * div_term) * gain
        pe = pe.squeeze(dim=1)
        return pe

    def position_initial(self):
        gain = 2.0 / (self.hidden_dim + 4.0 * self.hidden_dim)
        nn.init.xavier_normal_(self.positionEmbed.weight, gain=gain)

    def forward(self, position_ids: LongTensor):
        position_ids = position_ids.clamp_max(self.max_position-1)
        return self.positionEmbed(position_ids)

class SeqGNNNodeEmbedding(nn.Module):
    def __init__(self, pre_trained_name: str, oov_default='zero', pos_initial_type='sin_cos',
                 max_position: int = 15000, hidden_dim=300, add_position=True, word_emb_freeze=True):
        super(SeqGNNNodeEmbedding, self).__init__()
        self.add_position = add_position
        self.word_embedding = WordEmbedding(pre_trained_name=pre_trained_name,
                                            oov_default=oov_default, dim=hidden_dim, freeze=word_emb_freeze)
        if self.add_position:
            self.position_embedding = PositionEmbedding(max_position=max_position, hidden_dim=hidden_dim,
                                                        initial_type=pos_initial_type)

    def forward(self, input_ids, position_ids):
        inp_emb = self.word_embedding(input_ids)
        if self.add_position:
            pos_emb = self.position_embedding(position_ids)
            embeddings = inp_emb + pos_emb
        else:
            embeddings = inp_emb
        return embeddings



if __name__ == '__main__':
    # relEmbed = RelationEmbedding(num_relations=3)
    # linear_map = nn.Linear(in_features=300, out_features=512, bias=False)
    # x = linear_map(relEmbed.relEmbbed).view(-1, 8, 64)
    #
    # attn_r = nn.Parameter(torch.FloatTensor(1, 8, 64), requires_grad=True)
    #
    # er = (x * attn_r).sum(dim=-1).unsqueeze(-1)
    #
    # idx = torch.randint(0, 3, size=(30,))
    #
    # xx = er[idx]
    #
    # print(x.shape)
    # print(er.shape)
    # print(xx.shape)
    position_embed = PositionEmbedding(max_position=100, hidden_dim=300)

    for name, param in position_embed.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)

