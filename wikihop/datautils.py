from tqdm import tqdm
from core.embedding_utils import WordEmbedding
from wikihop.ioutils import load_gz_file, load_json_file
from nltk.tokenize import word_tokenize
import itertools
from itertools import accumulate
from nltk.tokenize import TweetTokenizer

def answer_id_extraction(answer, candidate_list):
    index = -1
    try:
        index = candidate_list.index(answer)
    except ValueError:
        print('{} is not is answer list'.format(answer))
    return index

def doc2sentence(doc: str, sent_tokenizer = None) -> list:
    if sent_tokenizer is None:
        from nltk.tokenize import sent_tokenize as sentence_tokener
        sent_tokenizer = sentence_tokener
    sentences = sent_tokenizer(doc)
    return [sentence for sentence in sentences if sentence is not None]

def doc_list_tokenizer(tokenizer, doc_list: list):
    doc_sentences = []
    doc_sent_tokens = []
    for doc in doc_list:
        doc_sentences_i = doc2sentence(doc)
        doc_sentences.append(doc_sentences_i)
        temp_sent_tokens = sentence_list_tokenizer(tokenizer, doc_sentences_i)
        doc_sent_tokens.append(temp_sent_tokens)
    return doc_sentences, doc_sent_tokens

def sentence_tokenizer(tokenizer, sentence: str):
    return tokenizer(sentence)

def sentence_list_tokenizer(tokenizer, sentence_list: list):
    return [tokenizer(sentence) for sentence in sentence_list]

def query_tokenizer(query: str, tokenizer):
    query_relation = query[0:query.index(' ')].strip()
    query_relation = query_relation.replace('_', ' ').strip()
    query_entity = query[query.index(' '):].strip()
    query_entity = query_entity.replace('_', ' ').strip()
    query_tokens = sentence_list_tokenizer(tokenizer=tokenizer, sentence_list=[query_relation, query_entity])
    return query_tokens

def candidates_tokenizer(tokenizer, candidates):
    return sentence_list_tokenizer(tokenizer=tokenizer, sentence_list=candidates)

def wikihop_example_extraction(data: list):
    # tokenizer = TweetTokenizer().tokenize
    tokenizer = word_tokenize
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

def wikihop_dump_features(example_file_name: str, word_embedder: WordEmbedding, add_special_token=True):
    def example2sequence(query_tokens, cands_tokens, supports_tokens):
        query_ids = word_embedder.decode_word2d_list(words_list=query_tokens) ## list of list[int]
        cands_ids = word_embedder.decode_word2d_list(words_list=cands_tokens) ## list of list[int]
        supps_ids = [word_embedder.decode_word2d_list(supp_doc) for supp_doc in supports_tokens] ## list of list of list[int]
        return query_ids, cands_ids, supps_ids

    examples = load_gz_file(example_file_name)
    feature_examples = []
    for example_id, example_values in tqdm(examples.items()):
        query_tokens, candidates_tokens, supports_tokens, answer_label_idx = example_values
        if add_special_token:
            #+++++++++++
            query_tokens[0] = [word_embedder.special_token_dict['query_start']] + query_tokens[0]
            query_tokens[-1] = query_tokens[-1] + [word_embedder.special_token_dict['query_end']]
            candidates_tokens = [[word_embedder.special_token_dict['entity_start']] + candidate +
                                 [word_embedder.special_token_dict['entity_end']] for candidate in candidates_tokens]
            for i in range(len(supports_tokens)):
                for j in range(len(supports_tokens[i])):
                    supports_tokens[i][j] = supports_tokens[i][j] + [word_embedder.special_token_dict['sep_token']]
            #+++++++++++
        query_ids, cands_ids, supps_ids = example2sequence(query_tokens=query_tokens, cands_tokens=candidates_tokens,
                                                           supports_tokens=supports_tokens)
        example_i = {'id': example_id, 'q_ids': query_ids, 'cand_ids': cands_ids, 'doc_ids': supps_ids, 'ans_idx': answer_label_idx}
        feature_examples.append(example_i)
    return feature_examples

def wikihop2dictionary(json_data_file_name):
    data = load_json_file(file_name=json_data_file_name)
    example_dict = {}
    for case in tqdm(data):
        example_dict[case['id']] = {key: value for key, value in case.items() if key not in {'id'}}
    return example_dict