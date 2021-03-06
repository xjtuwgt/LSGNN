from core.seqgnn_tokenizer import SegGNNTokenizer
from wikihop.ioutils import load_gz_file
from wikihop.datautils import answer_id_extraction, doc2sentence
from tqdm import tqdm

def load_wikihop_tokenizer(pretrained_file_name: str='allenai/longformer-base-4096'):
    special_tokens_dict = {'additional_special_tokens': ['<e>','</e>']}
    sg_tokenizer = SegGNNTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_file_name)
    sg_tokenizer.add_special_tokens(special_tokens_dict=special_tokens_dict)
    return sg_tokenizer

def query_tokenizer(query, tokenizer: SegGNNTokenizer):
    query_relation = query[0:query.index(' ')].strip()
    query_relation = query_relation.replace('_', ' ').strip()
    query_entity = query[query.index(' '):].strip()
    query_entity = query_entity.replace('_', ' ').strip()
    query_tokens = sentence_list_tokenizer(tokenizer=tokenizer, sentence_list=[query_relation, query_entity])
    query_tokens[0] = [tokenizer.cls_token, tokenizer.bos_token] + query_tokens[0]
    query_tokens[-1] = query_tokens[-1] + [tokenizer.eos_token]
    return query_tokens

def candidates_tokenizer(candidates, tokenizer: SegGNNTokenizer):
    candidate_tokens = sentence_list_tokenizer(tokenizer=tokenizer, sentence_list=candidates)
    candidate_tokens = [['<e>'] + _ + ['</e>'] for _ in candidate_tokens]
    return candidate_tokens

def sentence_list_tokenizer(tokenizer: SegGNNTokenizer, sentence_list: list):
    return [tokenizer.tokenize(sentence) for sentence in sentence_list]

def doc_list_tokenizer(doc_list, tokenizer: SegGNNTokenizer):
    doc_sentences = []
    doc_sent_tokens = []
    for doc in doc_list:
        doc_sentences_i = doc2sentence(doc)
        doc_sentences.append(doc_sentences_i)
        temp_sent_tokens = sentence_list_tokenizer(tokenizer, doc_sentences_i)
        temp_sent_tokens = [_ + [tokenizer.sep_token] for _ in temp_sent_tokens]
        doc_sent_tokens.append(temp_sent_tokens)
    return doc_sentences, doc_sent_tokens

def wikihop_seg_gnn_example_extraction(data, tokenizer: SegGNNTokenizer):
    def row_process(row: dict):
        query, answer = row['query'], row['answer']
        candidates, supports = row['candidates'], row['supports']
        answer_label_idx = answer_id_extraction(answer, candidates)
        #====================================================================
        query_tokens = query_tokenizer(query=query, tokenizer=tokenizer)
        candidates_tokens = candidates_tokenizer(tokenizer=tokenizer, candidates=candidates)
        supports_sentences, supports_tokens = doc_list_tokenizer(tokenizer=tokenizer, doc_list=supports)
        assert len(supports_tokens) == len(supports)
        return query_tokens, candidates_tokens, supports_tokens, answer_label_idx
    examples = {}
    for case in tqdm(data):
        case_id = case['id']
        query_tokens, candidates_tokens, supports_tokens, answer_label_idx = row_process(row=case)
        examples[case_id] = (query_tokens, candidates_tokens, supports_tokens, answer_label_idx)
    return examples

def wikihop_seq_gnn_dump_features(example_file_name: str, tokenizer: SegGNNTokenizer):
    def example2sequence(query_tokens, cands_tokens, supports_tokens):
        def decode_word2d_list(words_list):
            return [tokenizer.convert_tokens_to_ids(_) for _ in words_list]
        query_ids = decode_word2d_list(words_list=query_tokens) ## list of list[int]
        cands_ids = decode_word2d_list(words_list=cands_tokens) ## list of list[int]
        supps_ids = [decode_word2d_list(supp_doc) for supp_doc in supports_tokens] ## list of list of list[int]
        return query_ids, cands_ids, supps_ids

    examples = load_gz_file(example_file_name)
    feature_examples = []
    for example_id, example_values in tqdm(examples.items()):
        query_tokens, candidates_tokens, supports_tokens, answer_label_idx = example_values
        query_ids, cands_ids, supps_ids = example2sequence(query_tokens=query_tokens, cands_tokens=candidates_tokens,
                                                           supports_tokens=supports_tokens)
        example_i = {'id': example_id, 'q_ids': query_ids, 'cand_ids': cands_ids, 'doc_ids': supps_ids, 'ans_idx': answer_label_idx}
        feature_examples.append(example_i)
    return feature_examples