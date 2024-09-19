import torch
import torch.nn as nn

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        # seq shape :torch.Size([1, 2708, 512])
        # return 1X512
        else:
            msk = torch.unsqueeze(msk, -1)
            # (-1)+input_dim+1
            return torch.sum(seq * msk, 1) / torch.sum(msk)

