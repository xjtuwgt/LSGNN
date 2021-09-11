from dataclasses import dataclass, field
import numpy as np
import random
from tqdm import tqdm
import torch
from typing import List, Tuple
from codes.seq2graph_utils import seq2graph
import dgl

import gzip
import pickle

from toy_findcat.dataset import TokenizedDataset, SentenceDropDataset
from toy_findcat.example import Sentence, ExampleWithSentences

@dataclass
class FindCatSentence(Sentence):
    pass

@dataclass
class FindCatExample(ExampleWithSentences):
    target_tokens: List[int]
    positions: List[int]
    window_size: int = 24
    label: int = 0

def contains_subsequence_old(target, sequence):
    if len(target) == 0:
        return True

    if target[0] not in sequence:
        return False
    else:
        for i in range(len(sequence) - len(target) + 1):
            if sequence[i] == target[0]:
                subresult = contains_subsequence(target[1:], sequence[i + 1:])
                if subresult:
                    return True
        return False


def contains_subsequence(target, sequence):
    if len(target) == 0:
        return True
    remaining = sequence
    matched = 0
    for t in target:
        idx = 0
        while idx < len(remaining) and remaining[idx] != t:
            idx += 1
        if idx >= len(remaining):
            return False
        else:
            matched += 1
            if matched == len(target):
                return True
            remaining = remaining[idx + 1:]

RESERVED_TOKENS = 10
VOCAB_SIZE = 100
PAD = 0
CLS = 1
SEP = 2
MASK = 3
VOCAB = tuple(range(RESERVED_TOKENS, RESERVED_TOKENS + VOCAB_SIZE))

def neg_example_generation(target_tokens, vocab, exam_seq_len):
    retval = random.choices(vocab, k=exam_seq_len)
    while contains_subsequence(target_tokens, retval):
        retval = random.choices(vocab, k=exam_seq_len)
    return retval

class FindCatDataset(TokenizedDataset):
    def __init__(self, tokenizer_class="bert-base-uncased",
                 total_examples=1000, seqlen=(500,), vocab=VOCAB,
                 target_tokens='cat', prob=0.5,
                 window_size=24,
                 seed=42, data_file_name=None):
        super().__init__(tokenizer_class=tokenizer_class)
        random.seed(seed)

        self.prob = prob
        self.seqlen = seqlen
        self.vocab = vocab
        self.target_tokens = [[ord(x) - ord('a') + RESERVED_TOKENS for x in y] for y in target_tokens.split('_')]
        self.total_examples = total_examples
        self.window_size = window_size
        if data_file_name is None:
            self.data = self.data_generation()
        else:
            self.data = self.load_data_from_file(data_file_name=data_file_name)

    def _generate_example(self):
        target = int(random.random() < self.prob)
        target_tokens = random.choice(self.target_tokens)
        ##=========
        exam_seq_len = np.random.choice(self.seqlen, 1)[0]
        ##=========
        positions = []
        retval = neg_example_generation(target_tokens=target_tokens, exam_seq_len=exam_seq_len, vocab=self.vocab)
        if target == 1:
            positions = sorted(random.choices(list(range(exam_seq_len)), k=len(target_tokens)))
            for p_i, p in enumerate(positions):
                retval[p] = target_tokens[p_i]
        return FindCatExample(
            tokenized_sentences=[FindCatSentence(sentence_idx=s_i, token_ids=[s]) for s_i, s in enumerate(retval)],
            target_tokens=target_tokens, window_size=self.window_size, positions=positions, label=target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def data_generation(self):
        data = [self._generate_example() for _ in tqdm(range(self.total_examples))]
        return data

    def save_data_into_file(self, data_file_name):
        with gzip.open(data_file_name, 'wb') as fout:
            pickle.dump(self.data, fout)
        print('save {} records into {}'.format(len(self.data), data_file_name))

    def load_data_from_file(self, data_file_name):
        with gzip.open(data_file_name, 'rb') as fin:
            data = pickle.load(fin)
        print('load {} records from {}'.format(len(data), data_file_name))
        return data

def find_cat_collate_fn(examples):
    batched_labels = np.zeros((len(examples),), dtype=np.int64)
    batched_graphs = []
    graph_cls_idxes = []
    cls_idx = 0
    graph_cls_idxes.append(cls_idx)
    for ex_i, ex in enumerate(examples):
        ex_input_ids = [s.token_ids[0] for s in ex.tokenized_sentences]
        target_ids = ex.target_tokens
        ex_sequence = [CLS] + target_ids + [SEP] + ex_input_ids
        ex_target_len = 2 + len(target_ids)
        ex_global_idx = [_ for _ in range(ex_target_len)]
        ex_graph = seq2graph(sequence=ex_sequence, start_offset=ex_target_len,
                             global_idx=ex_global_idx, window_size=ex.window_size)
        ex_seq_len = len(ex_sequence)
        cls_idx = cls_idx + ex_seq_len
        graph_cls_idxes.append(cls_idx)
        batched_graphs.append(ex_graph)
        batched_labels[ex_i] = ex.label
    batched_graph = dgl.batch(batched_graphs)
    retval = {
        'input_graph': batched_graph,
        'cls_idx': torch.LongTensor(graph_cls_idxes),
        'labels': torch.from_numpy(batched_labels)
    }
    return retval


# if __name__ == "__main__":
#
#     # dataset = FindCatDataset(total_examples=10)
#     #
#     # examples = dataset.data
#     #
#     # print(len(examples))
#     # # for x in examples:
#     # #     print(len(x.tokenized_sentences))
#     #
#     cached_examples_file = 'test.pkl.gz'
#     # # with gzip.open(cached_examples_file, 'wb') as fout:
#     # #     pickle.dump(examples, fout)
#     # # print('save {} records into {}'.format(len(examples), cached_examples_file))
#     #
#     # dataset.save_data_into_file(data_file_name=cached_examples_file)
#
#     dataset = FindCatDataset(data_file_name=cached_examples_file)
#
#
#     # true_count = 0
#     # count = 20000
#     # for _ in tqdm(range(count)):
#     #     x = random.choices(dataset.vocab, k=300)
# #         y = contains_subsequence(target=dataset.target_tokens, sequence=x)
# #         if y:
# #             true_count = true_count + 1
# #         # while y:
# #         #     x = random.choices(dataset.vocab, k=300)
# #         #     y = contains_subsequence(target=dataset.target_tokens, sequence=x)
# #         #     true_count = true_count + 1
# #     print(true_count * 1.0/count)
# #
# #     # dataset.data_generation()
# #     # sdrop_dataset = SentenceDropDataset(dataset, sent_drop_prob=.1,
# #     #                                     example_validate_fn=lambda ex: find_cat_validation_fn(ex, dataset=dataset))
# #     #
# #     # from tqdm import tqdm
# #     # from torch.utils.data import DataLoader
# #     #
# #     # dataloader = DataLoader(sdrop_dataset, batch_size=32, collate_fn=find_cat_collate_fn)
# #     #
# #     # for batch in tqdm(dataloader):
#     #     pass