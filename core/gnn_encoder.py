from torch import nn
from os.path import join
from core.embedding_utils import RelationEmbedding, SeqGNNNodeEmbedding, SeqGNNEmbeddings
from core.layers import RGDTLayer, GDTLayer
from core.layers import LSTMWrapper, TCNWrapper

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
        return node_embed

####################################################
#LSTM for sequence
####################################################
class LSTMGDTEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMGDTEncoder, self).__init__()
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
        self.lstm_encoder = LSTMWrapper(input_dim=self.config.word_emb_dim, n_layer=self.config.lstm_layers,
                                        hidden_dim=self.config.word_emb_dim, bidir=True)
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

    def forward(self, batch: dict):
        lstm_input = self.node_embedder.forward(input_ids=batch['seq_inputs'])
        lstm_output = self.lstm_encoder.forward(input=lstm_input,
                                                input_lengths=batch['seq_lens'])
        batch_size, batch_seq_len, out_dim = lstm_output.shape
        lstm_output = lstm_output.contiguous().view(batch_size * batch_seq_len, out_dim)
        lstm_mask = batch['seq_mask'].view(batch_size * batch_seq_len)
        graph = batch['graph']
        with graph.local_scope():
            node_embed = lstm_output[lstm_mask==1]
            edge_embed = self.edge_embedder.relEmbbed
            for layer_idx, layer in enumerate(self.gdt_layers):
                if layer_idx == 0:
                    node_embed = layer(graph, node_embed, edge_embed)
                else:
                    node_embed = layer(graph, node_embed)
        return node_embed

####################################################
#TCN for sequence
####################################################
class TCNGDTEncoder(nn.Module):
    def __init__(self, config):
        super(TCNGDTEncoder, self).__init__()
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
        self.tcn_encoder = TCNWrapper(c_in=self.config.word_emb_dim, layers=[self.config.tcn_hid_dim] * self.config.tcn_layers,
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
        return node_embed


####################################################
#TCN for sequence with Transformer tokenizer
####################################################
class SeqTCNGDTEncoder(nn.Module):
    def __init__(self, config):
        super(SeqTCNGDTEncoder, self).__init__()
        self.config = config
        self.word_embed_type = self.config.word_embed_type
        assert self.word_embed_type == 'seq_gnn'
        self.node_embedder = SeqGNNEmbeddings(vocab_size=self.config.seq_gnn_vocab_size,
                                              pad_token_id=self.config.seq_gnn_pad_id,
                                              hidden_size=self.config.seq_gnn_word_emb_dim)
        self.tcn_encoder = TCNWrapper(c_in=self.config.seq_gnn_word_emb_dim, layers=[self.config.tcn_hid_dim] * self.config.tcn_layers,
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

    def forward(self, batch: dict):
        tcn_input = self.node_embedder.forward(input_ids=batch['seq_inputs'])
        tcn_input = tcn_input.transpose(-2, -1).contiguous()
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
        return node_embed