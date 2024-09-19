import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        # 定义一个神经网络的线性层
        # in_ft输入神经元的个数
        # out_ft输出神经元的个数
        # batch_size X in_ft -> batch_size X out_ft
        self.act = nn.PReLU() # if act == 'prelu' else act
        # self.act=nn.Tanh()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
            # spmm表示矩阵乘法
            # 参数0表示，如果第一维数为1，则删除
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
            # 偏置常数
        
        return self.act(out)  # 激活函数

