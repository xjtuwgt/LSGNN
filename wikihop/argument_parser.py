import os
import argparse
import torch
import json
import logging
import random
import numpy as np
from os.path import join
from envs import OUTPUT_FOLDER, HOME_DATA_FOLDER
from core.gpu_utils import get_single_free_gpu

logger = logging.getLogger(__name__)
def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def json_to_argv(json_file):
    j = json.load(open(json_file))
    argv = []
    for k, v in j.items():
        new_v = str(v) if v is not None else None
        argv.extend(['--' + k, new_v])
    return argv

def set_seed(args):
    ##+++++++++++++++++++++++
    random_seed = args.seed + args.local_rank
    ##+++++++++++++++++++++++
    seed_everything(seed=random_seed)

def seed_everything(seed: int) -> int:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed

def complete_default_parser(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.customer:
        args.fasttext_model = args.fasttext_model + '.wikihop'
        args.glove_model = args.glove_model + '.wikihop'
    # set n_gpu
    # ++++++++++++++
    if args.relative_position:
        args.num_relations = 2 * args.window_size + 1
        args.add_position = False
    #+++++++++++++++
    if HOME_DATA_FOLDER.startswith('/dfs/scratch0'):
        args.stanford = 'true'
    if args.local_rank == -1:
        if args.stanford:
            if torch.cuda.is_available():
                gpu_idx, _ = get_single_free_gpu()
                device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.data_parallel:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # output dir name
    if not args.exp_name:
        args.exp_name = '_'.join(['lr.' + str(args.learning_rate),
                          'bs.' + str(args.train_batch_size) + 'emb.' + args.word_embed_type])
    args.exp_name = os.path.join(args.output_dir, args.exp_name)
    set_seed(args)
    os.makedirs(args.exp_name, exist_ok=True)
    torch.save(args, join(args.exp_name, "training_args.bin"))
    return args


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None,
                        help="If set, this will be used as directory name in OUTOUT folder")
    parser.add_argument("--config_file",
                        type=str,
                        default=None,
                        help="configuration file for command parser")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_FOLDER,
                        help='Directory to save model and summaries')
    parser.add_argument('--db_path', type=str, default=HOME_DATA_FOLDER)
    parser.add_argument('--train_name', type=str, default='train.json')
    parser.add_argument('--dev_name', type=str, default='dev.json')
    parser.add_argument('--token_train_name', type=str, default='train.examples.pkl.gz')
    parser.add_argument('--token_dev_name', type=str, default='dev.examples.pkl.gz')
    parser.add_argument('--decode_train_name', type=str, default='train.features.pkl.gz')
    parser.add_argument('--decode_dev_name', type=str, default='dev.features.pkl.gz')
    parser.add_argument('--fasttext_model', type=str, default='wiki-news-300d-1M.vec')
    parser.add_argument('--glove_model', type=str, default='glove.840B.300d')
    parser.add_argument('--word_embed_type', type=str, default='fasttext')
    parser.add_argument('--customer', type=boolean_string, default='true')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--relation_emb_gamma', type=float, default=0.1)
    parser.add_argument('--relation_emb_dim', type=int, default=300)
    parser.add_argument('--word_emb_dim', type=int, default=300)
    parser.add_argument('--max_position', type=int, default=15000)
    parser.add_argument('--word_emb_freeze', type=boolean_string, default='true')
    parser.add_argument('--relative_position', type=boolean_string, default='true')
    parser.add_argument('--position_initial_type', type=str, default='sin_cos')
    parser.add_argument('--add_position', type=boolean_string, default='true')
    parser.add_argument('--embedding_dropout', type=float, default=0.25)
    parser.add_argument('--oov_type', type=str, default='rand')
    parser.add_argument('--num_relations', type=int, default=3)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--sent_drop_prob', type=float, default=0.1)
    parser.add_argument('--feat_drop', type=float, default=0.25)
    parser.add_argument('--attn_drop', type=float, default=0.25)
    parser.add_argument('--residual', type=boolean_string, default='true')
    parser.add_argument('--diff_head_tail', type=boolean_string, default='false')
    parser.add_argument('--ppr_diff', type=boolean_string, default='true')
    parser.add_argument('--stanford', type=boolean_string, default='true')

    parser.add_argument('--hop_num', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--layers', type=int, default=3)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=8)
    # Environment+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--cpu_num', type=int, default=8)
    parser.add_argument("--data_parallel",
                        default='false',
                        type=boolean_string,
                        help="use data parallel or not")
    parser.add_argument("--gpu_id", default=None, type=str, help="GPU id")
    parser.add_argument('--fp16',
                        type=boolean_string,
                        default='false',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # learning and log ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=20,
                        help="Log every X updates steps.")
    parser.add_argument('--eval_interval_ratio', type=float, default=0.25,
                        help="evaluate every X updates steps.")

    parser.add_argument("--optimizer", type=str, default="RAdam", choices=["AdamW", "RAdam"],
                        help="Choose the optimizer to use. Default RecAdam.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restart"],
                        help="Choose the optimizer to use. Default RecAdam.")
    parser.add_argument("--debug", type=boolean_string, default='false')

    return parser