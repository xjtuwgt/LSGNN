from core.gnn_encoder import SeqEncoder
from torch import nn
from core.layernorm_utils import ScaleNorm as layer_norm
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from core.optimizer_utils import RAdam
from pytorch_lightning.core.lightning import LightningModule
from wikihop.lossutils import ce_loss_computation as loss_function

class LightingSeqGNNWikiHopModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = SeqEncoder(config=config)
        self.final_layer_norm = layer_norm(self.config.hidden_dim)
        self.answer_prediction_layer = nn.Linear(in_features = self.config.hidden_dim, out_features=1)

    def forward(self, batch: dict):
        node_embed = self.encoder.forward(batch=batch)
        node_embed = self.final_layer_norm(node_embed)
        cand_ans_start_emb = node_embed[batch['cand_start']]
        cand_ans_end_emb = node_embed[batch['cand_end']]
        cand_ans_emb = (cand_ans_start_emb + cand_ans_end_emb) * 0.5
        cand_ans_scores = self.answer_prediction_layer(cand_ans_emb)
        return cand_ans_scores

    def training_step(self, batch, batch_idx):
        scores = self.forward(batch=batch)
        loss_log = loss_function(scores=scores, batch=batch)
        self.log("train_loss", loss_log['loss'])
        return loss_log['loss']

    def validation_step(self, batch, batch_idx):
        scores = self.forward(batch=batch)

    def test_step(self, batch, batch_idx):
        scores = self.forward(batch=batch)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (not any(nd in n for nd in no_decay))],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0,
            }
        ]
        if self.config.optimizer == 'RAdam':
            optimizer = RAdam(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        if self.config.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=self.config.total_steps)
        elif self.config.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=self.config.total_steps)
        elif self.config.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.config.warmup_steps,
                                                                           num_training_steps=self.config.total_steps)
        else:
            raise '{} is not supported'.format(self.config.lr_scheduler)
        return [optimizer], [scheduler]
# #########################################################################
class SeqGNNWikiHopModel(nn.Module):
    def __init__(self, config):
        super(SeqGNNWikiHopModel, self).__init__()
        self.config = config
        self.encoder = SeqEncoder(config=config)
        self.final_layer_norm = layer_norm(self.config.hidden_dim)
        self.answer_prediction_layer = nn.Linear(in_features = self.config.hidden_dim, out_features=1)

    def forward(self, batch: dict):
        node_embed = self.encoder.forward(batch=batch)
        node_embed = self.final_layer_norm(node_embed)
        cand_ans_start_emb = node_embed[batch['cand_start']]
        cand_ans_end_emb = node_embed[batch['cand_end']]
        cand_ans_emb = (cand_ans_start_emb + cand_ans_end_emb) * 0.5
        cand_ans_scores = self.answer_prediction_layer(cand_ans_emb)
        return cand_ans_scores

    def fixed_learning_rate_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (not any(nd in n for nd in no_decay))],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0,
            }
        ]
        if self.config.optimizer == 'RAdam':
            optimizer = RAdam(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        if self.config.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=self.config.total_steps)
        elif self.config.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=self.config.total_steps)
        elif self.config.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.config.warmup_steps,
                                                                           num_training_steps=self.config.total_steps)
        else:
            raise '{} is not supported'.format(self.config.lr_scheduler)
        return optimizer, scheduler