import torch.nn.functional as F
from torch import nn

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