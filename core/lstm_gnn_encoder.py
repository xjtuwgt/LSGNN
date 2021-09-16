from torch import nn
from os.path import join
from core.embedding_utils import RelationEmbedding, SeqGNNNodeEmbedding
from core.layernorm_utils import ScaleNorm as layer_norm
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from core.optimizer_utils import RAdam
from core.gnn_encoder import RGDTLayer, GDTLayer
from core.layers import LSTMWrapper

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
        self.lstm_encoder = LSTMWrapper(input_dim=self.config.word_emb_dim, n_layer=self.config.lstm_layers,
                                        hidden_dim=self.config.word_emb_dim, bidir=True)
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
        self.gdt_layers.append(module=RGDTLayer(in_ent_feats=2 * self.config.word_emb_dim,
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
        lstm_input = self.node_embedder.forward(input_ids=batch['seq_inputs'])
        lstm_output = self.lstm_encoder.forward(input=lstm_input,
                                                input_lengths=batch['seq_lens'])
        batch_size, batch_seq_len, out_dim = lstm_output.shape
        lstm_output = lstm_output.view(batch_size * batch_seq_len, out_dim)
        lstm_mask = batch['seq_mask'].view(batch_size * batch_seq_len)
        graph = batch['graph']
        with graph.local_scope():
            # inp_ids = graph.ndata['n_type']
            # pos_ids = graph.ndata['n_id']
            # node_embed = self.node_embedder(inp_ids, pos_ids)
            node_embed = lstm_output[lstm_mask==1]
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
        return optimizer, schedulerv