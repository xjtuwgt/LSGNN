from torch.utils.data import Dataset
from itertools import accumulate
import itertools
import random
from scipy.stats import beta
from codes.seq2graph_utils import seq2graph
import torch.nn.functional as F
import torch
import dgl

IGNORE_INDEX = -100

def span_generation(len_list):
    accumul_sum_list = list(accumulate(len_list))
    span_list = []
    for i in range(len(len_list) - 1):
        # span_list.append((accumul_sum_list[i], accumul_sum_list[i+1]))
        span_list.append((accumul_sum_list[i], accumul_sum_list[i + 1] - 1))
    return span_list

def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx
    return len(spans)

def example2sequence(query_ids, cands_ids, supps_ids, max_seq_len:int=None):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    query_ids = list(itertools.chain(*query_ids))
    seq_input_ids = query_ids
    query_len = len(query_ids)
    cand_len_list = [query_len]
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    query_span = (0, query_len - 1)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for cands in cands_ids:
        seq_input_ids += cands
        cand_len_list.append(len(cands))
    cand_spans = span_generation(len_list=cand_len_list)
    assert len(cand_spans) == len(cands_ids)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    query_cand_len = sum(cand_len_list)
    assert query_cand_len == (cand_spans[-1][1] + 1)
    sent_len_list = [query_cand_len]
    doc_len_list = [query_cand_len]
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for supps in supps_ids:
        doc_i_lens = []
        for sent_ids in supps:
            seq_input_ids += sent_ids
            doc_i_lens.append(len(sent_ids))
        sent_len_list += doc_i_lens
        doc_len_list.append(sum(doc_i_lens))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    assert len(sent_len_list) >= 2 and len(doc_len_list) >=2, 'sent number {}, ' \
                                                              'document number = {}, supp docs {}'.format(len(sent_len_list),
                                                                                            len(doc_len_list), supps_ids)
    sent_spans = span_generation(len_list=sent_len_list)
    doc_spans = span_generation(len_list=doc_len_list)
    assert sent_spans[-1][1] == doc_spans[-1][1]
    assert len(doc_spans) == len(supps_ids)
    # res = {'q_span': query_span, 'c_span': cand_spans, 's_span': sent_spans, 'd_span': doc_spans,
    #        'input_ids': seq_input_ids, 'q_len': query_len, 'q_c_len': query_cand_len}
    if max_seq_len is not None:
        if len(seq_input_ids) > max_seq_len:
            sent_max_index = _largest_valid_index(sent_spans, max_seq_len - 1)
            sent_spans = sent_spans[:sent_max_index]
            max_tok_length = sent_spans[-1][1]
            doc_max_index = _largest_valid_index(doc_spans, max_tok_length)
            doc_spans = doc_spans[:doc_max_index]
            seq_input_ids = seq_input_ids[:(max_seq_len-1)] + [seq_input_ids[-1]]
        assert len(seq_input_ids) <= max_seq_len
    return query_span, cand_spans, seq_input_ids, query_cand_len

# def sent_list_dropout(supps_ids: list, drop_ratio: float):
#     drop_supps_ids = []
#     for supps in supps_ids:
#         drop_supps = []
#         for sent_ids in supps:
#             random_num = random.random()
#             if random_num >= drop_ratio:
#                 drop_supps.append(sent_ids)
#         if len(drop_supps) > 0:
#             drop_supps_ids.append(drop_supps)
#     if len(drop_supps_ids) <= 2: ## drop_supps_ids
#         return supps_ids
#     return drop_supps_ids
class BetaWikiHopSpanDrop:
    def __init__(self, drop_ratio=0.1, beta_drop_scale=1.0, reverse=False):
        self.drop_ratio = drop_ratio
        self.beta_drop_scale = beta_drop_scale
        a = max(1, self.drop_ratio / (1 - self.drop_ratio))
        b = max(1, (1 - self.drop_ratio) / self.drop_ratio)
        self.alpha = a * self.beta_drop_scale
        self.beta = b * self.beta_drop_scale
        if reverse:
            self.alpha = b * self.beta_drop_scale
            self.beta = a * self.beta_drop_scale

    def __call__(self, spans_list: list):
        if self.drop_ratio == 0.0:
            return spans_list
        keep_spans_list = []
        # drop_prob = beta.rvs(self.alpha, self.beta)
        for spans in spans_list:
            keep_spans = []
            drop_prob = beta.rvs(self.alpha, self.beta)
            for span in spans:
                random_num = random.random()
                if random_num >= drop_prob:
                    keep_spans.append(span)
            if len(keep_spans) > 0:
                keep_spans_list.append(keep_spans)
        if len(keep_spans_list) < 1:  ## drop_supps_ids
            return spans_list
        return keep_spans_list

class WikiHopSpanDrop:
    def __init__(self, drop_ratio=0.1):
        self.drop_ratio = drop_ratio

    def __call__(self, spans_list: list):
        if self.drop_ratio == 0.0:
            return spans_list
        keep_spans_list = []
        for spans in spans_list:
            keep_spans = []
            for span in spans:
                random_num = random.random()
                # print(random_num, self.drop_ratio)
                if random_num >= self.drop_ratio:
                    keep_spans.append(span)
            if len(keep_spans) > 0:
                keep_spans_list.append(keep_spans)
        if len(keep_spans_list) < 1:  ## drop_supps_ids
            return spans_list
        return keep_spans_list

class WikihopTrainDataSet(Dataset):
    def __init__(self, examples,
                 window_size: int,
                 relative_position: bool,
                 pad_id: int=None,
                 max_ans_num: int = 80,
                 max_seq_length: int = 4096,
                 sent_drop_prob=0.1,
                 beta_drop_scale=1.0):
        self.examples = examples
        self.sent_drop_prob = sent_drop_prob
        self.beta_drop_scale = beta_drop_scale
        self.span_drop_func = BetaWikiHopSpanDrop(drop_ratio=self.sent_drop_prob, beta_drop_scale=self.beta_drop_scale)
        self.window_size = window_size
        self.max_ans_num = max_ans_num
        self.max_seq_length = max_seq_length
        self.relative_position = relative_position
        self.pad_id = pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        example_id, answer_label_idx = example['id'], example['ans_idx']
        query_ids, cands_ids, supps_ids = example['q_ids'], example['cand_ids'], example['doc_ids']
        drop_supps_ids = self.span_drop_func(spans_list=supps_ids)
        # if self.sent_drop_prob > 0:
        #     a = max(1, self.sent_drop_prob / (1 - self.sent_drop_prob))
        #     b = max(1, (1 - self.sent_drop_prob) / self.sent_drop_prob)
        #     sent_drop_prob = beta.rvs(a * self.beta_drop_scale, b * self.beta_drop_scale)
        #     drop_supps_ids = sent_list_dropout(supps_ids=supps_ids, drop_ratio=sent_drop_prob)
        # else:
        #     drop_supps_ids = supps_ids
        query_span, cand_spans, seq_input_ids, query_cand_len = example2sequence(query_ids=query_ids, cands_ids=cands_ids,
                                                                                 supps_ids=drop_supps_ids, max_seq_len=self.max_seq_length)
        cand_ans_num = len(cand_spans)
        ans_mask = [1] * cand_ans_num
        if cand_ans_num < self.max_ans_num:
            pad_ans_num = self.max_ans_num - cand_ans_num
            cand_spans += [(0,0)] * pad_ans_num
            ans_mask += [0] * pad_ans_num
        ans_start_pos = [_[0] for _ in cand_spans]
        ans_end_pos = [_[1] for _ in cand_spans]
        ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_start_pos = torch.LongTensor([query_span[0]])
        query_end_pos = torch.LongTensor([query_span[1]])
        ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        global_ids = [_ for _ in range(query_cand_len)]
        graph = seq2graph(sequence=seq_input_ids, start_offset=query_cand_len, global_idx=global_ids,
                          window_size=self.window_size, position=self.relative_position)
        # graph = seq2graph(sequence=seq_input_ids, start_offset=0, global_idx=global_ids,
        #                   window_size=self.window_size, position=self.relative_position)
        ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        cand_start_pos = torch.LongTensor(ans_start_pos)
        cand_end_pos = torch.LongTensor(ans_end_pos)
        cand_mask = torch.LongTensor(ans_mask)
        answer_label = torch.zeros(self.max_ans_num, dtype=torch.long)
        answer_label[answer_label_idx] = 1
        answer_label_id = torch.LongTensor([answer_label_idx])

        res = {'q_start': query_start_pos, 'q_end': query_end_pos, 'cand_start': cand_start_pos, 'cand_end': cand_end_pos, 'cand_mask': cand_mask,
               'label': answer_label, 'graph': graph, 'label_id': answer_label_id, 'id': example_id}
        if self.pad_id is not None:
            res['pad_id'] = self.pad_id
        return res


class WikihopDevDataSet(Dataset):
    def __init__(self, examples,
                 window_size: int,
                 relative_position: bool,
                 pad_id: int=None,
                 max_ans_num: int = 80,
                 max_seq_length: int = 4096):
        self.examples = examples
        self.window_size = window_size
        self.max_ans_num = max_ans_num
        self.max_seq_length = max_seq_length
        self.relative_position = relative_position
        self.pad_id = pad_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        example_id, answer_label_idx = example['id'], example['ans_idx']
        query_ids, cands_ids, supps_ids = example['q_ids'], example['cand_ids'], example['doc_ids']
        query_span, cand_spans, seq_input_ids, query_cand_len = example2sequence(query_ids=query_ids, cands_ids=cands_ids,
                                                                                 supps_ids=supps_ids, max_seq_len=self.max_seq_length)
        cand_ans_num = len(cand_spans)
        ans_mask = [1] * cand_ans_num
        if cand_ans_num < self.max_ans_num:
            pad_ans_num = self.max_ans_num - cand_ans_num
            cand_spans += [(0,0)] * pad_ans_num
            ans_mask += [0] * pad_ans_num
        ans_start_pos = [_[0] for _ in cand_spans]
        ans_end_pos = [_[1] for _ in cand_spans]
        ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_start_pos = torch.LongTensor([query_span[0]])
        query_end_pos = torch.LongTensor([query_span[1]])
        ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        global_ids = [_ for _ in range(query_cand_len)]
        graph = seq2graph(sequence=seq_input_ids, start_offset=query_cand_len, global_idx=global_ids,
                                  window_size=self.window_size, position=self.relative_position)
        # graph = seq2graph(sequence=seq_input_ids, start_offset=0, global_idx=global_ids,
        #                           window_size=self.window_size, position=self.relative_position)
        ## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        cand_start_pos = torch.LongTensor(ans_start_pos)
        cand_end_pos = torch.LongTensor(ans_end_pos)
        cand_mask = torch.LongTensor(ans_mask)
        answer_label = torch.zeros(self.max_ans_num, dtype=torch.long)
        answer_label[answer_label_idx] = 1
        answer_label_id = torch.LongTensor([answer_label_idx])
        res = {'q_start': query_start_pos, 'q_end': query_end_pos, 'cand_start': cand_start_pos, 'cand_end': cand_end_pos,
               'cand_mask': cand_mask,
               'label': answer_label, 'label_id': answer_label_id, 'graph': graph, 'id': example_id}
        if self.pad_id is not None:
            res['pad_id'] = self.pad_id
        return res

def collate_fn(data):
    graph_node_num_list = [_['graph'].number_of_nodes() for _ in data]
    graph_node_num_cum_list = list(accumulate(graph_node_num_list))
    batch_size = len(data)
    if batch_size > 1:
        for idx in range(1, batch_size):
            data[idx]['cand_start'] = data[idx]['cand_start'] + graph_node_num_cum_list[idx-1]
            data[idx]['cand_end'] = data[idx]['cand_end'] + graph_node_num_cum_list[idx-1]

            data[idx]['q_start'] = data[idx]['q_start'] + graph_node_num_cum_list[idx-1]
            data[idx]['q_end'] = data[idx]['q_end'] + graph_node_num_cum_list[idx-1]

    batch_cand_start = torch.stack([_['cand_start'] for _ in data])
    batch_cand_end = torch.stack([_['cand_end'] for _ in data])
    batch_cand_mask = torch.stack([_['cand_mask'] for _ in data])
    batch_ans_label = torch.stack([_['label'] for _ in data])
    batch_graph = dgl.batch([_['graph'] for _ in data])
    batch_ids = [_['id'] for _ in data]
    batch_ans_label_id = torch.stack([_['label_id'] for _ in data])
    batch_query_start = torch.stack([_['q_start'] for _ in data])
    batch_query_end = torch.stack([_['q_end'] for _ in data])
    return {'cand_start': batch_cand_start, 'cand_end': batch_cand_end, 'cand_mask': batch_cand_mask,
            'q_start': batch_query_start, 'q_end': batch_query_end,
           'label': batch_ans_label, 'graph': batch_graph, 'label_id': batch_ans_label_id, 'id': batch_ids}

def graph_collate_fn(data):
    graph_node_num_list = [_['graph'].number_of_nodes() for _ in data]
    seq_lengths = torch.as_tensor(graph_node_num_list, dtype=torch.int32)
    seq_lengths, arg_sort_idxes = torch.sort(seq_lengths, descending=True)
    data = [data[_] for _ in arg_sort_idxes.tolist()]
    graph_node_num_cum_list = list(accumulate(seq_lengths.tolist()))
    batch_size = len(data)
    if batch_size > 1:
        for idx in range(1, batch_size):
            data[idx]['cand_start'] = data[idx]['cand_start'] + graph_node_num_cum_list[idx-1]
            data[idx]['cand_end'] = data[idx]['cand_end'] + graph_node_num_cum_list[idx-1]

            data[idx]['q_start'] = data[idx]['q_start'] + graph_node_num_cum_list[idx-1]
            data[idx]['q_end'] = data[idx]['q_end'] + graph_node_num_cum_list[idx-1]
    batch_cand_start = torch.stack([_['cand_start'] for _ in data])
    batch_cand_end = torch.stack([_['cand_end'] for _ in data])
    batch_cand_mask = torch.stack([_['cand_mask'] for _ in data])
    batch_ans_label = torch.stack([_['label'] for _ in data])
    batch_graph = dgl.batch([_['graph'] for _ in data])
    batch_ids = [_['id'] for _ in data]
    batch_ans_label_id = torch.stack([_['label_id'] for _ in data])
    batch_query_start = torch.stack([_['q_start'] for _ in data])
    batch_query_end = torch.stack([_['q_end'] for _ in data])
    return {'cand_start': batch_cand_start, 'cand_end': batch_cand_end, 'cand_mask': batch_cand_mask,
            'q_start': batch_query_start, 'q_end': batch_query_end,
           'label': batch_ans_label, 'label_id': batch_ans_label_id,  'graph': batch_graph, 'id': batch_ids}

def graph_seq_collate_fn(data):
    graph_node_num_list = [_['graph'].number_of_nodes() for _ in data]
    seq_lengths = torch.as_tensor(graph_node_num_list, dtype=torch.int32)
    seq_lengths, arg_sort_idxes = torch.sort(seq_lengths, descending=True)
    data = [data[_] for _ in arg_sort_idxes.tolist()]
    graph_node_num_cum_list = list(accumulate(seq_lengths.tolist()))
    max_seq_len = seq_lengths[0].data.item()
    sequences = [data[0]['graph'].ndata['n_type']]
    batch_size = len(data)
    batch_seq_mask = torch.ones((batch_size, max_seq_len), dtype=torch.int32)
    if batch_size > 1:
        for idx in range(1, batch_size):
            data[idx]['cand_start'] = data[idx]['cand_start'] + graph_node_num_cum_list[idx-1]
            data[idx]['cand_end'] = data[idx]['cand_end'] + graph_node_num_cum_list[idx-1]
            data[idx]['q_start'] = data[idx]['q_start'] + graph_node_num_cum_list[idx-1]
            data[idx]['q_end'] = data[idx]['q_end'] + graph_node_num_cum_list[idx-1]
            pad_len = max_seq_len - data[idx]['graph'].number_of_nodes()
            seq_i = data[idx]['graph'].ndata['n_type']
            seq_i = F.pad(seq_i, (0, pad_len), value=data[idx]['pad_id'])
            sequences.append(seq_i)
            batch_seq_mask[idx][data[idx]['graph'].number_of_nodes():]=0
    batch_sequence = torch.stack(sequences)
    batch_seq_lens = seq_lengths
    batch_cand_start = torch.stack([_['cand_start'] for _ in data])
    batch_cand_end = torch.stack([_['cand_end'] for _ in data])
    batch_cand_mask = torch.stack([_['cand_mask'] for _ in data])
    batch_ans_label = torch.stack([_['label'] for _ in data])
    batch_graph = dgl.batch([_['graph'] for _ in data])
    batch_ids = [_['id'] for _ in data]
    batch_ans_label_id = torch.stack([_['label_id'] for _ in data])
    batch_query_start = torch.stack([_['q_start'] for _ in data])
    batch_query_end = torch.stack([_['q_end'] for _ in data])
    return {'cand_start': batch_cand_start, 'cand_end': batch_cand_end, 'cand_mask': batch_cand_mask,
            'q_start': batch_query_start, 'q_end': batch_query_end, 'seq_inputs': batch_sequence, 'seq_mask': batch_seq_mask,
            'seq_lens': batch_seq_lens, 'label': batch_ans_label, 'label_id': batch_ans_label_id,
            'graph': batch_graph, 'id': batch_ids}