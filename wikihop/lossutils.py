from torch import Tensor as T
from torch import nn
import torch
from torch.autograd import Variable
from wikihop.wikihopdataset import IGNORE_INDEX

def bce_loss_computation(scores: T, batch: dict, args=None):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    logits_aux = Variable(scores.data.new(scores.size(0), scores.size(1), 1).zero_())
    predictions = torch.cat([logits_aux, scores], dim=-1).contiguous()
    labels = batch['label']
    labels[batch['cand_mask'] == 0] = IGNORE_INDEX
    predictions = predictions.view(-1, 2)
    labels = labels.view(-1)
    loss = criterion.forward(predictions, labels)
    log = {
        'loss': loss
    }
    return log

def ce_loss_computation(scores: T, batch: dict, args=None):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    scores = scores.squeeze(-1)
    scores[batch['cand_mask'] == 0] = -1e30
    labels = batch['label_id']
    labels = labels.squeeze(-1)
    loss = criterion.forward(scores, labels)
    log = {
        'loss': loss
    }
    return log