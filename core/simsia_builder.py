import torch.nn as nn
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, base_encoder_out_dim, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        # create the encoder = base_encoder + a 3-layer projector
        self.prev_dim = base_encoder_out_dim
        self.encoder = nn.Sequential(base_encoder,
                                        nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                                        nn.BatchNorm1d(self.prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                                        nn.BatchNorm1d(self.prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(self.prev_dim, self.prev_dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of input
            x2: second views of input
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()


if __name__ == '__main__':
    print('*' * 30)
    base_encoder = nn.Linear(100, 300)
    pretrained_encoder = SimSiam(base_encoder, 300)

    print(pretrained_encoder)
