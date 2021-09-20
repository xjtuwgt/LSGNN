from wikihop.argument_parser import default_parser, complete_default_parser, json_to_argv
from wikihop.datahelper import DataHelper
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
from wikihop.wikihop_model import SeqGNNWikiHopModel
from wikihop.modelutils import model_parameter_summary
from wikihop.lossutils import ce_loss_computation as loss_function
# from wikihop.lossutils import bce_loss_computation as loss_function
from wikihop.modelutils import wikihop_model_evaluation

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
for key, value in vars(args).items():
    logging.info('{}\t{}'.format(key, value))
logger.info("IN CMD MODE")
logger.info("PyTorch version = {}".format(torch.__version__))
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_helper = DataHelper(config=args)
dev_data_loader = data_helper.wikihop_val_dataloader
train_data_loader = data_helper.wikihop_train_dataloader
# #########################################################################
if args.total_steps > 0:
    t_total_steps = args.total_steps
    args.num_train_epochs = args.max_steps // (len(train_data_loader) // args.gradient_accumulation_steps) + 1
else:
    t_total_steps = len(train_data_loader) // args.gradient_accumulation_steps * args.num_train_epochs
args.total_steps = t_total_steps
# ##########################################################################
model = SeqGNNWikiHopModel(config=args)
model.to(args.device)
# #########################################################################
# # Show model information
# #########################################################################
logging.info('Model Parameter Configuration:')
for name, param in model.named_parameters():
    logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
logging.info('*' * 75)
model_para_number = model_parameter_summary(m=model, only_trainable=True)
logging.info('Number of parameters of model = {}'.format(model_para_number))
logging.info('*' * 75)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #########################################################################
# # Get Optimizer
optimizer, scheduler = model.fixed_learning_rate_optimizers()
# ##########################################################################
if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    models, optimizer = amp.initialize([model], optimizer, opt_level=args.fp16_opt_level)
    assert len(models) == 1
    model = models[0]
# Distributed training (should be after apex fp16 initialization)
if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)
# #########################################################################
###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
total_batch_num = len(train_data_loader)
logger.info('Total number of batches = {}'.format(total_batch_num))
eval_batch_interval_num = int(total_batch_num * args.eval_interval_ratio) + 1
logger.info('Evaluate the model by = {} batches'.format(eval_batch_interval_num ))
# #########################################################################
global_step = 0
start_epoch = 0
best_accuracy = 0.0
best_metrics = None
best_predictions = None
best_model_name = None
training_logs = []
if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(args.exp_name)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# train_example_dict = wikihop2dictionary(json_data_file_name=join(args.db_path, 'wikihop', args.train_name))
# fasttext = WordEmbedding(pre_trained_name=join(args.db_path, 'models', args.fasttext_model))
# idx2word = {v: k for k, v in fasttext.word2idx.items()}
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
train_iterator = trange(start_epoch, start_epoch+int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
for epoch in train_iterator:
    epoch_iterator = tqdm(train_data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        model.train()
        for key, value in batch.items():
            if key not in ['id']:
                batch[key] = value.to(args.device)
        # ==================================================================
        scores = model(batch)
        loss_log = loss_function(scores=scores, batch=batch)
        # ==================================================================
        if args.n_gpu > 1:
            for key, value in loss_log.items():
                loss_log[key] = value.mean()
            loss = loss_log['loss']  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss_log['loss'] / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss_log['loss'], optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss_log['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        log = dict([(k, v.detach().item()) for k, v in loss_log.items()])
        training_logs.append(log)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                del training_logs
                gc.collect()
                training_logs = []
                logging.info('Train model evaluation at step_{}/epoch_{}'.format(global_step + 1, epoch + 1))
                for key, value in metrics.items():
                    logging.info('Metric {}: {:.5f}'.format(key, value))
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', loss_log['loss'], global_step)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if args.total_steps > 0 and global_step > args.total_steps:
            epoch_iterator.close()
            break
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        del batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if (step + 1) % eval_batch_interval_num == 0:
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                metrics, predictions = wikihop_model_evaluation(args=args, model=model, dataloader=dev_data_loader)
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_metrics = metrics
                    best_predictions = predictions
                    for key, value in metrics.items():
                        logging.info('Current Metric {}: {:.5f}'.format(key, value))
                        tb_writer.add_scalar(key, value, epoch)
            for key, value in best_metrics.items():
                logging.info('Best Metric {}: {:.5f}'.format(key, value))
                tb_writer.add_scalar(key, value, epoch)

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        metrics, predictions = wikihop_model_evaluation(args=args, model=model, dataloader=dev_data_loader)
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_metrics = metrics
            best_predictions = predictions
            for key, value in best_metrics.items():
                logging.info('Best Metric {}: {:.5f}'.format(key, value))
                tb_writer.add_scalar(key, value, epoch)

for key, val in best_metrics.items():
    logger.info("The best performance: {} = {:.5f}".format(key, val))

for key, val in vars(args).items():
    print('{}\t{}'.format(key, val))

if args.local_rank in [-1, 0]:
    tb_writer.close()