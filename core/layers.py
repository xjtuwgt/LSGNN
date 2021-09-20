import torch.nn.functional as F
from torch import nn
import torch
from dgl.nn.pytorch.utils import Identity
# from torch.nn import LayerNorm as layer_norm
from core.layernorm_utils import ScaleNorm as layer_norm
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from torch import Tensor
from torch.nn.utils import rnn
from torch.autograd import Variable
from torch import Tensor as T
from torch.nn.utils import weight_norm

def small_init_gain(d_in, d_out):
    return 2.0/(d_in + 4.0 * d_out)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.model_dim = model_dim
        self.hidden_dim = d_hidden
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def init(self):
        # gain = nn.init.calculate_gain('relu')
        gain = small_init_gain(d_in=self.model_dim, d_out=self.hidden_dim)
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        gain = small_init_gain(d_in=self.hidden_dim, d_out=self.model_dim)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++V
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

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class LSTMWrapper(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layer:int, concat=False, bidir=True, dropout=0.3, return_last=True):
        """
        :param input_dim:
        :param hidden_dim:
        :param n_layer:
        :param concat:
        :param bidir: bi-direction
        :param dropout:
        :param return_last:
        """
        super(LSTMWrapper, self).__init__()
        self.rnns = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                input_dim_ = input_dim
                output_dim_ = hidden_dim
            else:
                input_dim_ = hidden_dim if not bidir else hidden_dim * 2
                output_dim_ = hidden_dim
            self.rnns.append(nn.LSTM(input_dim_, output_dim_, 1, bidirectional=bidir, batch_first=True))
        self.dropout = dropout
        self.concat = concat
        self.n_layer = n_layer
        self.return_last = return_last

    def forward(self, input: T, input_lengths: T=None):
        # input_length must be in decreasing order if input_lengths is not none
        bsz, slen = input.shape[0], input.shape[1]
        output = input
        outputs = []
        for i in range(self.n_layer):
            output = F.dropout(output, p=self.dropout, training=self.training)
            if input_lengths is not None:
                lens = input_lengths.data.cpu().numpy()
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, _ = self.rnns[i](output)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Flatten()
    def forward(self, x):
        return self.flatten(self.gap(x))

class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()
        # print(stride, padding, dilation, ni, nf, ks)
        self.conv1 = weight_norm(nn.Conv1d(ni, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(nf, nf, ks, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(ni,nf,1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: batch_size X embeded_dim X seqlen
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TemporalConvNet(nn.Module):
    def __init__(self, c_in: int, layers: list, ks=2, dropout=0.):
        super(TemporalConvNet, self).__init__()
        self.tcn_layers = nn.ModuleList()
        for i in range(len(layers)):
            dilation_size = 2 ** i
            ni = c_in if i == 0 else layers[i - 1]
            nf = layers[i]
            self.tcn_layers.append(TemporalBlock(ni, nf, ks, stride=1, dilation=dilation_size,
                                                 padding=(ks - 1) * dilation_size,
                                                 dropout=dropout))
    def forward(self, x):
        for layer in self.tcn_layers:
            x = layer(x)
        return x

class TCNWrapper(nn.Module):
    def __init__(self, c_in: int, c_out: int, layers=[64] * 2, ks=7, conv_dropout=0., fc_dropout=0.):
        super(TCNWrapper, self).__init__()
        self.encoder = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(-2, -1)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)

class TCN(nn.Module):
    def __init__(self, c_in: int, c_out: int, layers=[64] * 2, ks=7, conv_dropout=0., fc_dropout=0.):
        super(TCN, self).__init__()
        self.encoder = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)