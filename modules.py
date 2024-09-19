
import torch
import config as CFG
from torchvision import models
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator
from sklearn.decomposition import PCA

def dim_reduce(List,data,method="PCA"):
    gene_dict={}
    pca = PCA(n_components=CFG.spot_embedding)
    pca.fit(data.cpu())
    res = pca.transform(data.cpu())
    res = torch.tensor(res, dtype=torch.float32).cuda()
    torch.save(res, 'pca_res.pt')
    t = 0
    for i in range(len(List)):
        temp_data = res[t:t + List[i], :]
        t += List[i]
        gene_dict[i]=temp_data

    return gene_dict



class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
    def forward(self, seq1, seq2, adj, sparse, msk=None):
        #         features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None
        # msk, samp_bias1, samp_bias2 全为NONE
        h_1 = self.gcn(seq1, adj, sparse)
        c = self.read(h_1,msk)
        c = self.sigm(c)
        h_2 = self.gcn(seq2, adj, sparse)
        # h_2 shape :torch.Size([1, 2708, 512])
        ret = self.disc(c, h_1, h_2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        #c = self.read(h_1, msk)
        return h_1 # c.detach()

    def get_loss(self,h_1,h_2,c):
        c=self.sigm(c)

        ret=self.disc(c,h_1,h_2)
        nb_nodes = h_1.shape[1]
        b_xtent = nn.BCEWithLogitsLoss()
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        lbl=lbl.cuda()
        loss = b_xtent(ret, lbl)
        return loss

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, embedding_dim=CFG.image_embedding):
        super().__init__()

        self.resnet50_feature_extractor = models.resnet50(pretrained=True)  # 导入
        self.reset_weight(embedding_dim)

    def reset_weight(self, embedding_dim=CFG.image_embedding):
        self.resnet50_feature_extractor.fc = nn.Linear(2048, embedding_dim)
        for param in self.resnet50_feature_extractor.parameters():
            param.requires_grad = True  # trainable

    def forward(self, x):
        return self.resnet50_feature_extractor(x)


class ImageEncoder_resnet50(nn.Module):

    def __init__(self, embedding_dim=CFG.image_embedding):
        super().__init__()

        self.resnet50_feature_extractor = models.resnet50(pretrained=True)  # 导入
        self.reset_weight(embedding_dim)

    def reset_weight(self, embedding_dim=CFG.image_embedding):
        self.resnet50_feature_extractor.fc = nn.Linear(2048, embedding_dim)
        for name,param in self.resnet50_feature_extractor.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad=True
            else:
                param.requires_grad=False

    def forward(self, x):
        # img1 = transform1(x)  # 对图片进行transform1的各种操作
        return self.resnet50_feature_extractor(x)

class ProjectionHead(nn.Module):
    def __init__(
            self,
    embedding_dim,
    projection_dim=CFG.projection_dim,
    dropout=CFG.dropout
    ):
        super().__init__()
        self.projection=nn.Linear(embedding_dim,projection_dim)
        self.gelu=nn.GELU()
        self.fc=nn.Linear(projection_dim,projection_dim)
        self.dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(projection_dim)

    def forward(self,x):
        projected=self.projection(x)
        x=self.gelu(projected)
        x=self.fc(x)
        x=self.dropout(x)
        x=x+projected
        x=self.layer_norm(x)
        return x

class ProjectionHead_image(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        # self.tanh=nn.Tanh()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        # x=nn.functional.normalize(x,dim=1)
        x = self.layer_norm(x)  # normalize
        return x