import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        # n_h=512
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c.shape:1X512
        c_x = torch.unsqueeze(c, 1)
        # 增加一个维度
        # 变成1X1X512
        c_x = c_x.expand_as(h_pl)
        # h_pl shape:1X2705X512

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        # c_x:torch.Size([1, 2708, 512])
        # sc_1:torch.Size([1, 2708])
        # self.f_k(h_pl, c_x) 的具体算法见参考资料
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        # shape:1X5416
        return logits

