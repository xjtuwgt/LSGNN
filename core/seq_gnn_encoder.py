from torch import nn
from core.embedding_utils import RelationEmbedding, SeqGNNEmbeddings
from core.layernorm_utils import ScaleNorm as layer_norm
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
from core.optimizer_utils import RAdam
from core.layers import RGDTLayer, GDTLayer
from core.layers import TCNWrapper

class SeqTCNGDTEncoder(nn.Module):
    def __init__(self, config):
        super(SeqTCNGDTEncoder, self).__init__()
        self.config = config
        self.word_embed_type = self.config.word_embed_type
        assert self.word_embed_type == 'seq_gnn'
        self.node_embedder = SeqGNNEmbeddings(vocab_size=self.config.seq_gnn_vocab_size,
                                              pad_token_id=self.config.seq_gnn_pad_id,
                                              hidden_size=self.config.seq_gnn_word_hidden_dim)
        self.tcn_encoder = TCNWrapper(c_in=self.config.seq_gnn_word_hidden_dim, layers=[self.config.tcn_hid_dim] * self.config.tcn_layers,
                                        c_out=self.config.tcn_out_dim, conv_dropout=self.config.tcn_conv_drop,
                                      fc_dropout=self.config.tcn_fc_drop, ks=self.config.tcn_kernel_size)

        self.edge_embedder = RelationEmbedding(num_relations=self.config.num_relations,
                                               dim=self.config.relation_emb_dim,
                                               gamma=self.config.relation_emb_gamma)
        self.gdt_layers = nn.ModuleList()
        self.gdt_layers.append(module=RGDTLayer(in_ent_feats=self.config.tcn_out_dim,
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
        tcn_input = self.node_embedder.forward(input_ids=batch['seq_inputs'])
        tcn_input = tcn_input.transpose(-2, -1)
        tcn_output = self.tcn_encoder(tcn_input)
        batch_size, batch_seq_len, out_dim = tcn_output.shape
        tcn_output = tcn_output.contiguous().view(batch_size * batch_seq_len, out_dim)
        tcn_mask = batch['seq_mask'].view(batch_size * batch_seq_len)
        graph = batch['graph']
        with graph.local_scope():
            node_embed = tcn_output[tcn_mask==1]
            edge_embed = self.edge_embedder.relEmbbed
            for layer_idx, layer in enumerate(self.gdt_layers):
                if layer_idx == 0:
                    node_embed = layer(graph, node_embed, edge_embed)
                else:
                    node_embed = layer(graph, node_embed)
            node_embed = self.final_layer_norm(node_embed)
            cand_ans_start_emb = node_embed[batch['cand_start']]
            cand_ans_end_emb = node_embed[batch['cand_end']]
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