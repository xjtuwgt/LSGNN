from os.path import join
from wikihop.ioutils import load_gz_file, load_train_wikihop_data, load_dev_wikihop_data
from wikihop.datautils import wikihop_example_extraction, wikihop_dump_features
from core.pretrianed_word_embed_store import restore_fasttext_pretrained, restore_glove_pretrained
import gzip, pickle
from wikihop.argument_parser import boolean_string
from envs import HOME_DATA_FOLDER, PRETRAINED_MODEL_FOLDER, PREPROCESS_FOLDER
import argparse
import itertools
from tqdm import tqdm
from core.embedding_utils import WordEmbedding
from wikihop.transformer_datautils import wikihop_seg_gnn_example_extraction, wikihop_seq_gnn_dump_features
from wikihop.transformer_datautils import load_wikihop_tokenizer

def wikihop_example_process(data_name, is_train_data=True):
    if is_train_data:
        data = load_train_wikihop_data(train_data_name=data_name)
    else:
        data = load_dev_wikihop_data(dev_data_name=data_name)
    examples = wikihop_example_extraction(data=data)
    return examples

def wikihop_example_seq_graph_process(data_name, tokenizer, is_train_data=True):
    if is_train_data:
        data = load_train_wikihop_data(train_data_name=data_name)
    else:
        data = load_dev_wikihop_data(dev_data_name=data_name)
    examples = wikihop_seg_gnn_example_extraction(data=data, tokenizer=tokenizer)
    return examples

def wikihop_train_dev_seq_graph_tokenize_examples(args):
    assert args.word_embed_type == 'seq_gnn'
    tokenizer = load_wikihop_tokenizer(pretrained_file_name=args.tokenizer_name)
    train_data_name = join(HOME_DATA_FOLDER, 'wikihop', args.train_name)
    processed_train_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.' + args.token_train_name)
    train_examples = wikihop_example_seq_graph_process(data_name=train_data_name, tokenizer=tokenizer, is_train_data=True)
    with gzip.open(processed_train_data_name, 'wb') as fout:
        pickle.dump(train_examples, fout)
    print('Saving {} examples in {}'.format(len(train_examples), processed_train_data_name))

    dev_data_name = join(HOME_DATA_FOLDER, 'wikihop', args.dev_name)
    processed_dev_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.' + args.token_dev_name)
    dev_examples = wikihop_example_seq_graph_process(data_name=dev_data_name, tokenizer=tokenizer, is_train_data=False)
    with gzip.open(processed_dev_data_name, 'wb') as fout:
        pickle.dump(dev_examples, fout)
    print('Saving {} examples in {}'.format(len(dev_examples), processed_dev_data_name))

def wikihop_train_dev_tokenize(args):
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 90)
    train_data_name = join(HOME_DATA_FOLDER, 'wikihop', args.train_name)
    processed_train_data_name = join(PREPROCESS_FOLDER, 'nltk.' + args.token_train_name)
    train_examples = wikihop_example_process(data_name=train_data_name, is_train_data=True)
    with gzip.open(processed_train_data_name, 'wb') as fout:
        pickle.dump(train_examples, fout)
    print('Saving {} examples in {}'.format(len(train_examples), processed_train_data_name))

    dev_data_name = join(HOME_DATA_FOLDER, 'wikihop', args.dev_name)
    processed_dev_data_name = join(PREPROCESS_FOLDER, 'nltk.' + args.token_dev_name)
    dev_examples = wikihop_example_process(data_name=dev_data_name, is_train_data=False)
    with gzip.open(processed_dev_data_name, 'wb') as fout:
        pickle.dump(dev_examples, fout)
    print('Saving {} examples in {}'.format(len(dev_examples), processed_dev_data_name))

    wikihop_vocab_dict = wikihop_vocab_collection(train_data=train_examples, dev_data=dev_examples)
    wikihop_vocab_file_name = join(PREPROCESS_FOLDER, args.vocab_name)
    with gzip.open(wikihop_vocab_file_name, 'wb') as fout:
        pickle.dump(wikihop_vocab_dict, fout)
    print('Saving {} examples in {}'.format(len(wikihop_vocab_dict), wikihop_vocab_file_name))


def wikihop_train_dev_dump_features(args, customer=True):
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 90)
    if args.word_embed_type == 'fasttext':
        if customer:
            word_embedder = WordEmbedding(pre_trained_name=join(PRETRAINED_MODEL_FOLDER, args.fasttext_model + '.wikihop'))
        else:
            word_embedder = WordEmbedding(pre_trained_name=join(PRETRAINED_MODEL_FOLDER, args.fasttext_model))
    elif args.word_embed_type == 'glove':
        if customer:
            word_embedder = WordEmbedding(pre_trained_name=join(PRETRAINED_MODEL_FOLDER, args.glove_model + '.wikihop'))
        else:
            word_embedder = WordEmbedding(pre_trained_name=join(PRETRAINED_MODEL_FOLDER, args.glove_model))
    else:
        raise 'wrong word embedding type = {}'.format(args.word_embed_type)
    train_example_file_name = join(PREPROCESS_FOLDER, 'nltk.' + args.token_train_name)
    if customer:
        processed_train_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.wikihop.' + args.decode_train_name)
    else:
        processed_train_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.' + args.decode_train_name)
    train_examples = wikihop_dump_features(example_file_name=train_example_file_name, word_embedder=word_embedder, add_special_token=args.add_special_token)
    with gzip.open(processed_train_data_name, 'wb') as fout:
        pickle.dump(train_examples, fout)
    print('Saving {} examples in {}'.format(len(train_examples), processed_train_data_name))

    dev_example_file_name = join(PREPROCESS_FOLDER, 'nltk.' + args.token_dev_name)
    if customer:
        processed_dev_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.wikihop.' + args.decode_dev_name)
    else:
        processed_dev_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.' + args.decode_dev_name)
    dev_examples = wikihop_dump_features(example_file_name=dev_example_file_name, word_embedder=word_embedder, add_special_token=args.add_special_token)
    with gzip.open(processed_dev_data_name, 'wb') as fout:
        pickle.dump(dev_examples, fout)
    print('Saving {} examples in {}'.format(len(dev_examples), processed_dev_data_name))


def wikihop_train_dev_seq_gnn_dump_features(args):
    assert args.word_embed_type == 'seq_gnn'
    tokenizer = load_wikihop_tokenizer(args.tokenizer_name)
    train_example_file_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.' +args.token_train_name)
    processed_train_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.wikihop.' + args.decode_train_name)
    train_dump_examples = wikihop_seq_gnn_dump_features(example_file_name=train_example_file_name, tokenizer=tokenizer)
    with gzip.open(processed_train_data_name, 'wb') as fout:
        pickle.dump(train_dump_examples, fout)
    print('Saving {} examples in {}'.format(len(train_dump_examples), processed_train_data_name))

    dev_example_file_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.' + args.token_dev_name)
    processed_dev_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.wikihop.' + args.decode_dev_name)
    dev_dump_examples = wikihop_seq_gnn_dump_features(example_file_name=dev_example_file_name, tokenizer=tokenizer)
    with gzip.open(processed_dev_data_name, 'wb') as fout:
        pickle.dump(dev_dump_examples, fout)
    print('Saving {} examples in {}'.format(len(dev_dump_examples), processed_dev_data_name))

def wikihop_vocab_collection(train_data: dict, dev_data: dict):
    def vocab_collection(data: dict, voc_dict: dict):
        for case_id, value in tqdm(data.items()):
            query_tokens, candidates_tokens, supports_tokens, _ = value
            query_tokens = list(itertools.chain(*query_tokens))
            candidates_tokens = list(itertools.chain(*candidates_tokens))
            for token in query_tokens:
                count = voc_dict.get(token, 0)
                voc_dict[token] = count + 1
            for token in candidates_tokens:
                count = voc_dict.get(token, 0)
                voc_dict[token] = count + 1
            for support_doc_tokens in supports_tokens:
                support_doc_tokens = list(itertools.chain(*support_doc_tokens))
                for token in support_doc_tokens:
                    count = voc_dict.get(token, 0)
                    voc_dict[token] = count + 1
        return voc_dict
    vocab_dict = {}
    vocab_dict = vocab_collection(data=train_data, voc_dict=vocab_dict)
    print('Vocab size over train data = {}'.format(len(vocab_dict)))
    vocab_dict = vocab_collection(data=dev_data, voc_dict=vocab_dict)
    print('Vocab size over dev data = {}'.format(len(vocab_dict)))
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return vocab_dict

def restore_fast_text_word_embeddings(args, customer=True):
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 90)
    fasttext_model_name = join(PRETRAINED_MODEL_FOLDER, args.fasttext_model)
    wikihop_vocab_dict = None
    if customer:
        vocab_file_name = join(PREPROCESS_FOLDER, args.vocab_name)
        wikihop_vocab_dict = load_gz_file(file_name=vocab_file_name)
        fasttext_model_dat_name = join(PRETRAINED_MODEL_FOLDER, args.fasttext_model + '.wikihop.dat')
        fasttext_model_dat_word = join(PRETRAINED_MODEL_FOLDER, args.fasttext_model + '.wikihop.words.pkl')
        fasttext_model_dat_word2idx = join(PRETRAINED_MODEL_FOLDER, args.fasttext_model + '.wikihop.idx.pkl')
    else:
        fasttext_model_dat_name = join(PRETRAINED_MODEL_FOLDER, args.fasttext_model + '.dat')
        fasttext_model_dat_word = join(PRETRAINED_MODEL_FOLDER, args.fasttext_model + '.words.pkl')
        fasttext_model_dat_word2idx = join(PRETRAINED_MODEL_FOLDER, args.fasttext_model + '.idx.pkl')
    restore_fasttext_pretrained(pretrained_fast_text_file_name=fasttext_model_name,
                                dat_path_extend_name=fasttext_model_dat_name,
                                word_pickle_name=fasttext_model_dat_word,
                                word2index_name=fasttext_model_dat_word2idx,
                                customer_word_dict=wikihop_vocab_dict)

def restore_glove_word_embeddings(args, customer=True):
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 90)
    glove_model_name = join(PRETRAINED_MODEL_FOLDER, args.glove_model +'.txt')
    wikihop_vocab_dict = None
    if customer:
        vocab_file_name = join(PREPROCESS_FOLDER, args.vocab_name)
        wikihop_vocab_dict = load_gz_file(file_name=vocab_file_name)
        glove_model_dat_name = join(PRETRAINED_MODEL_FOLDER, args.glove_model +'.wikihop.dat')
        glove_model_dat_word = join(PRETRAINED_MODEL_FOLDER, args.glove_model + '.wikihop.words.pkl')
        glove_model_dat_word2idx = join(PRETRAINED_MODEL_FOLDER, args.glove_model + '.wikihop.idx.pkl')
    else:
        glove_model_dat_name = join(PRETRAINED_MODEL_FOLDER, args.glove_model +'.dat')
        glove_model_dat_word = join(PRETRAINED_MODEL_FOLDER, args.glove_model + '.words.pkl')
        glove_model_dat_word2idx = join(PRETRAINED_MODEL_FOLDER, args.glove_model + '.idx.pkl')
    restore_glove_pretrained(pretrained_glove_file_name=glove_model_name,
                             dat_path_extend=glove_model_dat_name,
                             word_pickle_name=glove_model_dat_word,
                             word2index_name=glove_model_dat_word2idx,
                             dim=300, customer_word_dict=wikihop_vocab_dict)

def wiki_hop_train_dev_seq_gnn_preprocess(args):
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 90)
    # wikihop_train_dev_seq_graph_tokenize_examples(args=args)
    wikihop_train_dev_seq_gnn_dump_features(args=args)

def wikihop_data_analysis(args):
    for key, value in vars(args).items():
        print('{}\t{}'.format(key, value))
    print('*' * 90)
    processed_train_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.' + args.decode_train_name)
    processed_dev_data_name = join(PREPROCESS_FOLDER, args.word_embed_type + '.' + args.decode_dev_name)
    train_data = load_gz_file(processed_train_data_name)
    dev_data = load_gz_file(processed_dev_data_name)

    cand_num_dict = {}
    supp_num_dict = {}

    sent_len_list = []

    for example in tqdm(train_data):
        cands = example['cand_ids']
        count = cand_num_dict.get(len(cands), 0)
        cand_num_dict[len(cands)] = count + 1
        supps = example['doc_ids']
        for supp in supps:
            for sent in supp:
                sent_len_list.append(len(sent))
        count = supp_num_dict.get(len(supps), 0)
        supp_num_dict[len(supps)] = count + 1


    for example in tqdm(dev_data):
        cands = example['cand_ids']
        count = cand_num_dict.get(len(cands), 0)
        cand_num_dict[len(cands)] = count + 1
        supps = example['doc_ids']
        for supp in supps:
            for sent in supp:
                sent_len_list.append(len(sent))
        count = supp_num_dict.get(len(supps), 0)
        supp_num_dict[len(supps)] = count + 1

    for key in sorted(cand_num_dict):
        print(key, cand_num_dict[key])

    for key in sorted(supp_num_dict):
        print(key, supp_num_dict[key])

    print(sum(sent_len_list)/len(sent_len_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=HOME_DATA_FOLDER)
    parser.add_argument('--train_name', type=str, default='train.json')
    parser.add_argument('--dev_name', type=str, default='dev.json')
    parser.add_argument('--token_train_name', type=str, default='train.examples.pkl.gz')
    parser.add_argument('--token_dev_name', type=str, default='dev.examples.pkl.gz')
    parser.add_argument('--vocab_name', type=str, default='wikihop.vocab.pkl.gz')
    parser.add_argument('--decode_train_name', type=str, default='train.features.pkl.gz')
    parser.add_argument('--decode_dev_name', type=str, default='dev.features.pkl.gz')
    parser.add_argument('--fasttext_model', type=str, default='wiki-news-300d-1M.vec') #'wiki-news-300d-1M-subword.vec'
    parser.add_argument('--glove_model', type=str, default='glove.840B.300d')
    parser.add_argument('--word_embed_type', type=str, default='fasttext')
    parser.add_argument('--add_special_token', type=boolean_string, default='true')
    parser.add_argument('--tokenizer_name', type=str, default='allenai/longformer-base-4096')
    args = parser.parse_args()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ## Step 1: wiki-hop data tokenization
    # wikihop_train_dev_tokenize(args=args)
    #
    # ## Step 2: model preprocess
    # restore_fast_text_word_embeddings(args=args)
    # restore_glove_word_embeddings(args=args)

    # Step 3: dump-features: tokens map to ids
    # wikihop_train_dev_dump_features(args=args)
    # args.word_embed_type = 'glove'
    # wikihop_train_dev_dump_features(args=args)

    # # Step 4: data analysis
    args.word_embed_type = 'seq_gnn'
    wiki_hop_train_dev_seq_gnn_preprocess(args=args)

    # wikihop_data_analysis(args=args)