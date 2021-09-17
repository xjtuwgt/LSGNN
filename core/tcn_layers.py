from torch import nn
from torch.nn.utils import weight_norm

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
        self.conv1 = weight_norm(nn.Conv1d(ni,nf,ks,stride=stride,padding=padding,dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(nf,nf,ks,stride=stride,padding=padding,dilation=dilation))
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
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

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

class TCN(nn.Module):
    def __init__(self, c_in: int, c_out:int, layers=[25] * 2, ks=7, conv_dropout=0., fc_dropout=0.):
        super(TCN, self).__init__()
        self.encoder = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1],c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)

if __name__  == '__main__':
    model = TCN(c_in=300, c_out=20)
    print('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)
    import torch
    x = torch.rand((2, 300, 1000))
    y = model(x)
    print(y.shape)