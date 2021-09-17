import torch
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from torch.autograd import Variable

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





if __name__ == '__main__':
    model = LSTMWrapper(input_dim=10, hidden_dim=20, n_layer=2)
    print('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)
    input = torch.rand(2, 20, 10)
    lengths = torch.tensor([20, 10], dtype=torch.int32)
    print(lengths.shape)
    print(input.shape)
    output = model(input, lengths)
    print(output.shape)