from torch import nn
from os.path import join
from core.embedding_utils import RelationEmbedding, SeqGNNNodeEmbedding
import torch
from dgl.nn.pytorch.utils import Identity
# from torch.nn import LayerNorm as layer_norm
from core.utils import small_init_gain
from core.layernorm_utils import ScaleNorm as layer_norm
from core.utils import PositionwiseFeedForward
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from torch import Tensor
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from core.optimizer_utils import RAdam

####################################################
#Pre-layer norm + small initialization, https://github.com/tnq177/transformers_without_tears
####################################################

class GDTEncoder(nn.Module):
    def __init__(self, config):
        super(GDTEncoder, self).__init__()
        self.config = config
        self.word_embed_type = self.config.word_embed_type
        if self.word_embed_type == 'glove':
            self.word_embed_file_name = self.config.glove_model
        elif self.word_embed_type == 'fasttext':
            self.word_embed_file_name = self.config.fasttext_model
        else:
            raise '{} is not supported'.format(self.word_embed_file_name)
        self.node_embedder = SeqGNNNodeEmbedding(pre_trained_name=join(self.config.db_path, 'models', self.word_embed_file_name),
                                             oov_default=self.config.oov_type,
                                             hidden_dim=self.config.word_emb_dim,
                                             max_position=self.config.max_position,
                                             pos_initial_type=self.config.position_initial_type,
                                             add_position=self.config.add_position,
                                             word_emb_freeze=self.config.word_emb_freeze)

        self.edge_embedder = RelationEmbedding(num_relations=self.config.num_relations,
                                               dim=self.config.relation_emb_dim,
                                               gamma=self.config.relation_emb_gamma)
        self.gdt_layers = nn.ModuleList()
        self.gdt_layers.append(module=RGDTLayer(in_ent_feats=self.config.word_emb_dim,
                                               in_rel_feats=self.config.relation_emb_dim,
                                               out_ent_feats=self.config.hidden_dim,
                                               num_heads=self.config.head_num,
                                               hop_num=self.config.hop_num,
                                               alpha=self.config.alpha,
                                               feat_drop=self.config.feat_drop,
                                               attn_drop=self.config.attn_drop,
                                               residual=self.config.residual,
                                               diff_head_tail=self.config.diff_head_tail,
                                               ppr_diff=self.config.ppr_diff))
        for _ in range(1, self.config.layers):
            self.gdt_layers.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                               out_ent_feats=self.config.hidden_dim,
                                               num_heads=self.config.head_num,
                                               hop_num=self.config.hop_num,
                                               alpha=self.config.alpha,
                                               feat_drop=self.config.feat_drop,
                                               attn_drop=self.config.attn_drop,
                                               residual=self.config.residual,
                                               diff_head_tail=self.config.diff_head_tail,
                                               ppr_diff=self.config.ppr_diff))
        self.final_layer_norm = layer_norm(self.config.hidden_dim)
        self.answer_prediction_layer = nn.Linear(in_features= self.config.hidden_dim, out_features=1)

    def forward(self, batch: dict):
        graph = batch['graph']
        with graph.local_scope():
            inp_ids = graph.ndata['n_type']
            pos_ids = graph.ndata['n_id']
            node_embed = self.node_embedder(inp_ids, pos_ids)
            edge_embed = self.edge_embedder.relEmbbed
            for layer_idx, layer in enumerate(self.gdt_layers):
                if layer_idx == 0:
                    node_embed = layer(graph, node_embed, edge_embed)
                else:
                    node_embed = layer(graph, node_embed)
            node_embed = self.final_layer_norm(node_embed)
            cand_ans_start_emb = node_embed[batch['cand_start']+1]
            cand_ans_end_emb = node_embed[batch['cand_end']-1]
            # cand_ans_emb = torch.cat([cand_ans_start_emb, cand_ans_end_emb], dim=-1)
            cand_ans_emb = (cand_ans_start_emb + cand_ans_end_emb) * 0.5
            cand_ans_scores = self.answer_prediction_layer(cand_ans_emb)
        return cand_ans_scores

    def fixed_learning_rate_optimizers(self, total_steps):
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
                                                        num_training_steps=total_steps)
        elif self.config.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=total_steps)
        elif self.config.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.config.warmup_steps,
                                                                           num_training_steps=total_steps)
        else:
            raise '{} is not supported'.format(self.config.lr_scheduler)
        return optimizer, scheduler

#######################################################################################################################
#######################################################################################################################
class RGDTLayer(nn.Module):
    def __init__(self,
                 in_ent_feats: int,
                 in_rel_feats: int,
                 out_ent_feats: int,
                 num_heads: int,
                 hop_num: int,
                 alpha: float=0.15,
                 feat_drop: float=0.1,
                 attn_drop: float=0.1,
                 negative_slope=0.2,
                 residual=True,
                 activation=None,
                 diff_head_tail=False,
                 ppr_diff=True):
        super(RGDTLayer, self).__init__()
        self._in_head_ent_feats, self._in_tail_ent_feats = in_ent_feats, in_ent_feats
        self._out_ent_feats = out_ent_feats
        self._in_rel_feats = in_rel_feats
        self._num_heads = num_heads
        self._hop_num = hop_num
        self._alpha = alpha

        assert self._out_ent_feats % self._num_heads == 0
        self._head_dim = self._out_ent_feats // self._num_heads
        self.diff_head_tail = diff_head_tail

        if diff_head_tail: ## make different map
            self._ent_fc_head = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)
            self._ent_fc_tail = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)
        else:
            self._ent_fc = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)

        self._rel_fc = nn.Linear(self._in_rel_feats, self._num_heads * self._head_dim, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.attn_h = nn.Parameter(torch.FloatTensor(1, self._num_heads, self._head_dim), requires_grad=True)
        self.attn_t = nn.Parameter(torch.FloatTensor(1, self._num_heads, self._head_dim), requires_grad=True)
        self.attn_r = nn.Parameter(torch.FloatTensor(1, self._num_heads, self._head_dim), requires_grad=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope) ### for attention computation

        if residual:
            if in_ent_feats != out_ent_feats:
                self.res_fc_ent = nn.Linear(in_ent_feats, self._num_heads * self._head_dim, bias=False)
            else:
                self.res_fc_ent = Identity()
        else:
            self.register_buffer('res_fc_ent', None)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_layer_norm = layer_norm(in_ent_feats)
        self.ff_layer_norm = layer_norm(self._num_heads * self._head_dim)
        self.feed_forward_layer = PositionwiseFeedForward(model_dim=self._num_heads * self._head_dim,
                                                          d_hidden=4 * self._num_heads * self._head_dim)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.reset_parameters()
        self.activation = activation
        self.ppr_diff = ppr_diff

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        # gain = nn.init.calculate_gain('relu')
        gain = small_init_gain(d_in=self._in_head_ent_feats, d_out=self._in_head_ent_feats)
        if self.diff_head_tail:
            nn.init.xavier_normal_(self._ent_fc_head.weight, gain=gain)
            nn.init.xavier_normal_(self._ent_fc_tail.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self._ent_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self._rel_fc.weight, gain=gain)
        if isinstance(self.res_fc_ent, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_ent.weight, gain=gain)

    def forward(self, graph, ent_feat: Tensor, rel_feat: Tensor, get_attention=False):
        ent_head = ent_tail = self.graph_layer_norm(ent_feat)
        ent_head = self.feat_drop(ent_head)
        ent_tail = self.feat_drop(ent_tail)
        rel_emb = self.feat_drop(rel_feat)
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               ' Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')
            if self.diff_head_tail:
                feat_head = self._ent_fc_head(ent_head).view(-1, self._num_heads, self._head_dim)
                feat_tail = self._ent_fc_tail(ent_tail).view(-1, self._num_heads, self._head_dim)
            else:
                feat_head = feat_tail = self._ent_fc(ent_head).view(-1, self._num_heads, self._head_dim)
            feat_rel = self._rel_fc(rel_emb).view(-1, self._num_heads, self._head_dim)
            eh = (feat_head * self.attn_h).sum(dim=-1).unsqueeze(-1)
            et = (feat_tail * self.attn_t).sum(dim=-1).unsqueeze(-1)
            er = (feat_rel * self.attn_r).sum(dim=-1).unsqueeze(-1)
            ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            edge_ids = graph.edata['e_type']
            er = er[edge_ids]
            ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            graph.srcdata.update({'ft': feat_head, 'eh': eh})
            graph.dstdata.update({'et': et})
            graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
            e = self.leaky_relu(graph.edata.pop('e') + er)
            if self.ppr_diff:
                graph.edata['a'] = edge_softmax(graph, e)
                rst = self.ppr_estimation(graph=graph)
            else:
                graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                 fn.sum('m', 'ft'))
                rst = graph.dstdata['ft']
            # residual
            if self.res_fc_ent is not None:
                resval = self.res_fc_ent(ent_tail).view(ent_tail.shape[0], -1, self._head_dim)
                rst = self.feat_drop(rst) + resval  # residual
            rst = rst.flatten(1)
            # +++++++++++++++++++++++++++++++++++++++
            ff_rst = self.feed_forward_layer(self.feat_drop(self.ff_layer_norm(rst)))
            rst = self.feat_drop(ff_rst) + rst # residual
            # +++++++++++++++++++++++++++++++++++++++
            # activation
            if self.activation:
                rst = self.activation(rst)
            # +++++++++++++++++++++++++++++++++++++++
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

    def ppr_estimation(self, graph):
        graph = graph.local_var()
        feat_0 = graph.srcdata.pop('ft')
        feat = feat_0
        attentions = graph.edata.pop('a')
        for _ in range(self._hop_num):
            graph.srcdata['h'] = self.feat_drop(feat)
            graph.edata['a_temp'] = self.attn_drop(attentions)
            graph.update_all(fn.u_mul_e('h', 'a_temp', 'm'), fn.sum('m', 'h'))
            feat = graph.dstdata.pop('h')
            feat = (1.0 - self._alpha) * self.feat_drop(feat) + self._alpha * feat_0
        return feat

class GDTLayer(nn.Module):
    def __init__(self,
                 in_ent_feats: int,
                 out_ent_feats: int,
                 num_heads: int,
                 hop_num: int,
                 alpha: float=0.15,
                 feat_drop: float=0.1,
                 attn_drop: float=0.1,
                 negative_slope=0.2,
                 residual=True,
                 activation=None,
                 diff_head_tail=False,
                 ppr_diff=True):
        super(GDTLayer, self).__init__()
        self._in_head_ent_feats, self._in_tail_ent_feats = in_ent_feats, in_ent_feats
        self._out_ent_feats = out_ent_feats
        self._num_heads = num_heads
        self._hop_num = hop_num
        self._alpha = alpha

        assert self._out_ent_feats % self._num_heads == 0
        self._head_dim = self._out_ent_feats // self._num_heads
        self.diff_head_tail = diff_head_tail

        if diff_head_tail: ## make different map
            self._ent_fc_head = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)
            self._ent_fc_tail = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)
        else:
            self._ent_fc = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.attn_h = nn.Parameter(torch.FloatTensor(1, self._num_heads, self._head_dim), requires_grad=True)
        self.attn_t = nn.Parameter(torch.FloatTensor(1, self._num_heads, self._head_dim), requires_grad=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope) ### for attention computation

        if residual:
            if in_ent_feats != out_ent_feats:
                self.res_fc_ent = nn.Linear(in_ent_feats, self._num_heads * self._head_dim, bias=False)
            else:
                self.res_fc_ent = Identity()
        else:
            self.register_buffer('res_fc_ent', None)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_layer_norm = layer_norm(in_ent_feats)
        self.ff_layer_norm = layer_norm(self._num_heads * self._head_dim)
        self.feed_forward_layer = PositionwiseFeedForward(model_dim=self._num_heads * self._head_dim,
                                                          d_hidden=4 * self._num_heads * self._head_dim)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.reset_parameters()
        self.activation = activation
        self.ppr_diff = ppr_diff

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        # gain = nn.init.calculate_gain('relu')
        gain = small_init_gain(d_in=self._in_head_ent_feats, d_out=self._in_head_ent_feats)
        if self.diff_head_tail:
            nn.init.xavier_normal_(self._ent_fc_head.weight, gain=gain)
            nn.init.xavier_normal_(self._ent_fc_tail.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self._ent_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_h, gain=gain)
        nn.init.xavier_normal_(self.attn_t, gain=gain)
        if isinstance(self.res_fc_ent, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_ent.weight, gain=gain)

    def forward(self, graph, ent_feat: Tensor, get_attention=False):
        ent_head = ent_tail = self.graph_layer_norm(ent_feat)
        ent_head = self.feat_drop(ent_head)
        ent_tail = self.feat_drop(ent_tail)

        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               ' Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')
            if self.diff_head_tail:
                feat_head = self._ent_fc_head(ent_head).view(-1, self._num_heads, self._head_dim)
                feat_tail = self._ent_fc_tail(ent_tail).view(-1, self._num_heads, self._head_dim)
            else:
                feat_head = feat_tail = self._ent_fc(ent_head).view(-1, self._num_heads, self._head_dim)
            eh = (feat_head * self.attn_h).sum(dim=-1).unsqueeze(-1)
            et = (feat_tail * self.attn_t).sum(dim=-1).unsqueeze(-1)
            ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            graph.srcdata.update({'ft': feat_head, 'eh': eh})
            graph.dstdata.update({'et': et})
            graph.apply_edges(fn.u_add_v('eh', 'et', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            if self.ppr_diff:
                graph.edata['a'] = edge_softmax(graph, e)
                rst = self.ppr_estimation(graph=graph)
            else:
                graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                 fn.sum('m', 'ft'))
                rst = graph.dstdata['ft']
            # residual
            if self.res_fc_ent is not None:
                resval = self.res_fc_ent(ent_tail).view(ent_tail.shape[0], -1, self._head_dim)
                rst = self.feat_drop(rst) + resval  # residual
            rst = rst.flatten(1)
            # +++++++++++++++++++++++++++++++++++++++
            ff_rst = self.feed_forward_layer(self.feat_drop(self.ff_layer_norm(rst)))
            rst = self.feat_drop(ff_rst) + rst # residual
            # +++++++++++++++++++++++++++++++++++++++
            # +++++++++++++++++++++++++++++++++++++++
            # activation
            if self.activation:
                rst = self.activation(rst)
            # +++++++++++++++++++++++++++++++++++++++
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

    def ppr_estimation(self, graph):
        graph = graph.local_var()
        feat_0 = graph.srcdata.pop('ft')
        feat = feat_0
        attentions = graph.edata.pop('a')
        for _ in range(self._hop_num):
            graph.srcdata['h'] = self.feat_drop(feat)
            graph.edata['a_temp'] = self.attn_drop(attentions)
            graph.update_all(fn.u_mul_e('h', 'a_temp', 'm'), fn.sum('m', 'h'))
            feat = graph.dstdata.pop('h')
            feat = (1.0 - self._alpha) * self.feat_drop(feat) + self._alpha * feat_0
        return feat
