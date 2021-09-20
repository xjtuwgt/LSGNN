# # from codes.seq2graph_utils import seq2graph
# # import dgl
# # from tqdm import tqdm
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # def train_acc_computation(file_name):
# #     train_accs = []
# #     dev_accs = []
# #     with open(file_name) as f:
# #         lines = f.readlines()
# #         for line in lines:
# #             if 'Train accuracy' in line:
# #                 train_acc = line[(line.index('=') + 1):line.index('at')].strip()
# #                 train_acc = float(train_acc)
# #                 train_accs.append(train_acc)
# #                 # print(train_acc)
# #                 # print(line)
# #             if 'dev accuracy' in line:
# #                 dev_acc = line[(line.index('=') + 1):(line.index(', current'))].strip()
# #                 dev_acc = float(dev_acc)
# #                 dev_accs.append(dev_acc)
# #                 # print(dev_acc)
# #                 # print(line)
# #     print(len(train_accs))
# #     dev_accs = dev_accs[:len(train_accs)]
# #     print(len(dev_accs))
# #     # print(train_accs)
# #
# #     train_accs = np.array(train_accs)
# #     dev_accs = np.array(dev_accs)
# #     #
# #     # print(train_accs.shape, dev_accs.shape)
# #
# #     return train_accs, dev_accs
# #
# #
# # if __name__ == '__main__':
# #     N = 8192
# #     file_name = '/Users/xjtuwgt/Desktop/temp.txt'
# #     train_accs, dev_accs = train_acc_computation(file_name=file_name)
# #
# #     x = np.arange(len(train_accs)) + 1.0
# #     x = np.log(x)
# #     # print(x)
# #
# #     plt.plot(x, train_accs, label="train acc")
# #     plt.plot(x, dev_accs, label="dev acc")
# #     plt.show()
# #     # import random
# #     # x = [_ for _ in range(N)]
# #     # random.shuffle(x)
# #     # graph_list = []
# #     # for i in range(5):
# #     #     g_i = seq2graph(sequence=x, global_idx=[0], window_size=100, skip_step=20)
# #     #     graph_list.append(g_i)
# #     #
# #     # batch_graph = dgl.batch(graph_list)
# #     # print(batch_graph.ndata)
# #     # print(type(g))
# #     # print(g.number_of_nodes())
# #     # print(g.number_of_edges() * 1.0/(N * N))
# #     # print(g.ndata['n_type'])
# #     # g = seq2graph(sequence=x, global_idx=[0], window_size=100, skip_step=20)
# #     # print(len(y) /(4096 * 4096))
# #     # for _ in y:
# #     #     print(_)
#
#
# # from wikihop.argument_parser import default_parser, complete_default_parser
# # from wikihop.datahelper import DataHelper
# # import logging
# # import torch
# # from tqdm import tqdm, trange
# # from tensorboardX import SummaryWriter
# # from wikihop.datautils import wikihop2dictionary
# # from core.embedding_utils import WordEmbedding
# # from os.path import join
# # from time import time
# # from core.gnn_encoder import GDTEncoder
# # from wikihop.lossutils import ce_loss_computation
# # from wikihop.modelutils import wikihop_model_evaluation
# #
# # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
# #                     datefmt='%m/%d/%Y %H:%M:%S',
# #                     level=logging.INFO)
# # logger = logging.getLogger(__name__)
# # # #########################################################################
# # # # Initialize arguments
# # # ##########################################################################
# # parser = default_parser()
# # args = parser.parse_args()
# # args = complete_default_parser(args=args)
# # for key, value in vars(args).items():
# #     logging.info('{}\t{}'.format(key, value))
# # logger.info("IN CMD MODE")
# # logger.info("PyTorch version = {}".format(torch.__version__))
# # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # data_helper = DataHelper(config=args)
# # dev_data_loader = data_helper.wikihop_val_dataloader
# # train_data_loader = data_helper.wikihop_train_dataloader
# # ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # train_example_dict = wikihop2dictionary(json_data_file_name=join(args.db_path, 'wikihop', args.dev_name))
# # fasttext = WordEmbedding(pre_trained_name=join(args.db_path, 'models', args.fasttext_model))
# # idx2word = {v: k for k, v in fasttext.word2idx.items()}
# # idx2word[999994] = 'oov'
# # ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # for step, batch in enumerate(train_data_loader):
# #     for key, value in batch.items():
# #         if key not in ['id']:
# #             batch[key] = value.to(args.device)
# #         print(key, value)
# #     # ==================================================================
# #     graph_node_ids = batch['graph'].ndata['n_type']
# #     cand_start = batch['cand_start']
# #     cand_end = batch['cand_end']
# #     cand_mask = batch['cand_mask']
# #     label_id = batch['label_id'].tolist()
# #     query_start = batch['q_start'].tolist()
# #     query_end = batch['q_end'].tolist()
# #     # print(graph_node_ids)
# #     node_ids = graph_node_ids.tolist()
# #     for idx, _ in enumerate(batch['id']):
# #         query = train_example_dict[_]['query']
# #         candidates = train_example_dict[_]['candidates']
# #         answer = train_example_dict[_]['answer']
# #
# #         cand_start_i = cand_start[idx].tolist()
# #         cand_end_i = cand_end[idx].tolist()
# #         cand_mask_i = cand_mask[idx].tolist()
# #
# #         query_ids = node_ids[query_start[idx][0]:(query_end[idx][0] + 1)]
# #         decode_query = ' '.join([idx2word[c] for c in query_ids])
# #         print(decode_query)
# #         print(query)
# #
# #         for m_idx, m_i in enumerate(cand_mask_i):
# #             if m_i == 1:
# #                 start_i = cand_start_i[m_idx]
# #                 end_i = cand_end_i[m_idx]
# #                 cand_i = node_ids[start_i:(end_i + 1)]
# #                 cand_text_i = ' '.join([idx2word[c] for c in cand_i])
# #                 # print(m_idx, cand_text_i, '\t\t\t', candidates[m_idx])
# #         true_anwer = answer
# #         answer_id = label_id[idx][0]
# #         decoded_true_answer = node_ids[cand_start[idx][answer_id]: (cand_end[idx][answer_id] + 1)]
# #         decoded_true_answer = ' '.join([idx2word[c] for c in decoded_true_answer])
# #         print('*' * 100)
# #         # print(true_anwer, decoded_true_answer)
# #     break
#
# # from core.seqgnn_tokenizer import SegGNNTokenizer
# # from wikihop.transformer_datautils import load_wikihop_tokenizer
# # tokenizer_file_name = 'allenai/longformer-base-4096'
# # sg_tokenizer = SegGNNTokenizer.from_pretrained(tokenizer_file_name)
# #
# #
# # print(sg_tokenizer.special_tokens_map)
# # print(sg_tokenizer.vocab_size)
# #
# # wiki_sg_tokenizer = load_wikihop_tokenizer(pretrained_file_name=tokenizer_file_name)
# # print(wiki_sg_tokenizer.vocab_size)
# # print(wiki_sg_tokenizer.special_tokens_map)
#
# from wikihop.argument_parser import default_parser, complete_default_parser, json_to_argv
# from wikihop.datahelper import DataHelper
# import logging
# import torch
# from tqdm import tqdm, trange
# from tensorboardX import SummaryWriter
# import gc
# import sys
# from wikihop.datautils import wikihop2dictionary
# from core.embedding_utils import WordEmbedding
# from os.path import join
# from time import time
# from core.gpu_utils import get_single_free_gpu
# from core.gnn_encoder import GDTEncoder
# from wikihop.lossutils import ce_loss_computation as loss_function
# # from wikihop.lossutils import bce_loss_computation as loss_function
# from wikihop.modelutils import wikihop_model_evaluation
# from wikihop.modelutils import model_parameter_summary
#
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)
# # #########################################################################
# # # Initialize arguments
# # ##########################################################################
# parser = default_parser()
# logger.info("IN CMD MODE")
# logger.info("Pytorch version = {}".format(torch.__version__))
# args_config_provided = parser.parse_args(sys.argv[1:])
# if args_config_provided.config_file is not None:
#     argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
# else:
#     argv = sys.argv[1:]
# args = parser.parse_args(argv)
# # ##########################################################################
# args.word_embed_type = 'seq_gnn'
# args = complete_default_parser(args=args)
# for key, value in vars(args).items():
#     logging.info('{}\t{}'.format(key, value))
# logger.info("IN CMD MODE")
# logger.info("PyTorch version = {}".format(torch.__version__))
#
#
#

import torch
from time import time
start_time = time()
x = torch.rand((4, 4096, 200))
for i in range(100):
    y = x.transpose(-1, -2)
print(time() - start_time)

