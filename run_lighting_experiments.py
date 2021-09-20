from wikihop.argument_parser import default_parser, complete_default_parser, json_to_argv
from wikihop.datahelper import WikiHopDataModule
import logging
import torch
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import gc
import sys
from wikihop.datautils import wikihop2dictionary
from core.embedding_utils import WordEmbedding
from os.path import join
from time import time
from core.gpu_utils import get_single_free_gpu
from wikihop.wikihop_model import LightingSeqGNNWikiHopModel
from wikihop.modelutils import model_parameter_summary

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# #########################################################################
# # Initialize arguments
# ##########################################################################
parser = default_parser()
logger.info("IN CMD MODE")
logger.info("Pytorch version = {}".format(torch.__version__))
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
# ##########################################################################
args = complete_default_parser(args=args)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
wikihop_data_module = WikiHopDataModule(config=args)
wikihop_data_module.setup()
train_data_loader = wikihop_data_module.train_dataloader()
# #########################################################################
if args.total_steps > 0:
    t_total_steps = args.total_steps
    args.num_train_epochs = args.max_steps // (len(train_data_loader) // args.gradient_accumulation_steps) + 1
else:
    t_total_steps = len(train_data_loader) // args.gradient_accumulation_steps * args.num_train_epochs
args.total_steps = t_total_steps
# #########################################################################
for key, value in vars(args).items():
    logging.info('{}\t{}'.format(key, value))
logger.info("IN CMD MODE")
logger.info("PyTorch version = {}".format(torch.__version__))
# #########################################################################
light_model = LightingSeqGNNWikiHopModel(config=args)
# #########################################################################
# # Show model information
# #########################################################################
logging.info('Model Parameter Configuration:')
for name, param in light_model.named_parameters():
    logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
logging.info('*' * 75)
model_para_number = model_parameter_summary(m=light_model, only_trainable=True)
logging.info('Number of parameters of model = {}'.format(model_para_number))
logging.info('*' * 75)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++