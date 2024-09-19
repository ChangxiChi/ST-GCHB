import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import config as CFG
from modules import ImageEncoder, ProjectionHead, ProjectionHead_image,DGI,ImageEncoder_resnet50
import time
import random



def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)  # ln x
    loss = (-targets * log_softmax(preds)).sum(1)
    # loss = (-targets * preds).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class DGI_spot(nn.Module):
    def __init__(
            self,
            lr=0.001,
            l2_coef=0.0,
            hid_units=768,
            ft_size=CFG.spot_embedding,
            nonlinearity='prelu'
    ):
        super().__init__()
        self.model = DGI(ft_size, hid_units, nonlinearity)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr, weight_decay=l2_coef)
        # self.spot_projection = ProjectionHead(embedding_dim=2048)

    def forward(self, features, adj):
        nb_nodes = features.shape[0]
        b_xtent = nn.BCEWithLogitsLoss()
        self.optimiser.zero_grad()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[idx, :]
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        lbl = lbl.cuda()
        # features = torch.tensor(features)
        features = features.unsqueeze(0)
        # shuf_fts = torch.tensor(shuf_fts)
        shuf_fts = shuf_fts.unsqueeze(0)
        logits = self.model(features, shuf_fts, adj, sparse=1)
        loss = b_xtent(logits, lbl)
        # loss.backward()
        # self.optimiser.step()
        # self.optimiser.zero_grad()
        # torch.cuda.empty_cache()
        return loss


class DGI_model(nn.Module):
    def __init__(
            self,
            lr=0.001,
            l2_coef=0.0,
            hid_units=768,
            ft_size=CFG.spot_embedding,
            nonlinearity='prelu'
    ):
        super().__init__()
        self.model = DGI(ft_size, hid_units, nonlinearity)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr, weight_decay=l2_coef)
        # self.spot_projection = ProjectionHead(embedding_dim=2048)

    def forward(self, features, adj):
        nb_nodes = features.shape[0]
        b_xtent = nn.BCEWithLogitsLoss()
        self.optimiser.zero_grad()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[idx, :]
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        lbl = lbl.cuda()
        # features = torch.tensor(features)
        features = features.unsqueeze(0)
        # shuf_fts = torch.tensor(shuf_fts)
        shuf_fts = shuf_fts.unsqueeze(0)
        logits = self.model(features, shuf_fts, adj, sparse=1)
        loss = b_xtent(logits, lbl)
        loss.backward()
        self.optimiser.step()
        self.optimiser.zero_grad()
        return loss.item()


class DGI_image(nn.Module):
    def __init__(
            self,
            lr=0.001,
            l2_coef=0.0,
            hid_units=512,
            ft_size=2048,
            nonlinearity='prelu',
    ):
        super().__init__()
        self.model = DGI(ft_size, hid_units, nonlinearity)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr, weight_decay=l2_coef)
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=2048)

    def forward(self, batch, adj):
        for idx in range(batch.shape[0]):
            if idx == 0:
                features = batch[idx]
                features = features.unsqueeze(0)
                features = self.image_encoder(features)
            else:
                embedding = batch[idx]
                embedding = embedding.unsqueeze(0)
                embedding = self.image_encoder(embedding)
                features = torch.cat([features, embedding])
            torch.cuda.empty_cache()
        features = features.cuda()
        # features = self.image_encoder(batch)
        nb_nodes = features.shape[0]
        self.optimiser.zero_grad()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[idx, :]
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        adj = adj.cuda()
        logits = self.model(features, shuf_fts, adj, sparse=1)
        b_xtent = nn.BCEWithLogitsLoss()
        lbl = lbl.cuda()

        loss = b_xtent(logits, lbl)
        loss.backward()
        self.optimiser.step()
        torch.cuda.empty_cache()
        return loss.item()






class ST_GCHB(nn.Module):
    def __init__(
            self,
            embedding_dim_spot=256,
            embedding_dim_image=256,
            ft_size_spot=813,  # dim after DR
            ft_size_image=1000,  # dim after resnet50
            projection_dim=128,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet50().cuda()
        self.spot_projection = ProjectionHead(embedding_dim=embedding_dim_spot, projection_dim=projection_dim).cuda()
        self.image_projection = ProjectionHead_image(embedding_dim=embedding_dim_image,
                                                     projection_dim=projection_dim).cuda()
        self.spot_dgi_model = DGI_model(ft_size=ft_size_spot, hid_units=embedding_dim_spot).cuda()
        self.image_dgi_model = DGI_model(ft_size=ft_size_image, hid_units=embedding_dim_image).cuda()

    def forward(self, graph_batch, gene_fts, adj, optimizer):
        idx = list(range(0, graph_batch.shape[0]))
        random_seed = int(time.time())
        random.seed = random_seed
        random.shuffle(idx)
        batch_size = 16
        for i in range(int((graph_batch.shape[0] + 1) / batch_size) + 1):
            for x in range(int((graph_batch.shape[0] + 1) / batch_size) + 1):
                if x == 0:
                    graph_fts = self.image_encoder(graph_batch[batch_size * x:batch_size * x + batch_size])
                elif batch_size * x + batch_size - 1 <= graph_batch.shape[0]:
                    t = self.image_encoder(graph_batch[batch_size * x:batch_size * x + batch_size])
                    graph_fts = torch.cat([graph_fts, t])
                else:
                    t = self.image_encoder(graph_batch[batch_size * x:gene_fts.shape[0]])
                    graph_fts = torch.cat([graph_fts, t])
            gene_embed = self.spot_dgi_model.model.embed(gene_fts, adj, 1, None).squeeze()
            graph_embed = self.image_dgi_model.model.embed(graph_fts, adj, 1, None).squeeze()

            if batch_size * i + batch_size - 1 <= graph_batch.shape[0]:
                spot_features = gene_embed[idx[batch_size * i:batch_size * (i + 1)]]
                spot_Input = gene_fts[idx[batch_size * i:batch_size * (i + 1)]]
                graph_features = graph_embed[idx[batch_size * i:batch_size * (i + 1)]]
                graph_Input = graph_fts[idx[batch_size * i:batch_size * (i + 1)]]

            else:
                spot_features = gene_embed[idx[batch_size * i:gene_fts.shape[0]]]
                spot_Input = gene_fts[idx[batch_size * i:gene_fts.shape[0]]]
                graph_features = graph_embed[idx[batch_size * i:gene_fts.shape[0]]]
                graph_Input = graph_fts[idx[batch_size * i:gene_fts.shape[0]]]
            optimizer.zero_grad()
            spot_embeddings = self.spot_projection(spot_features)
            graph_embeddings = self.image_projection(graph_features)
            HSIC_loss_spot = nHSIC(spot_Input, spot_features) - CFG.Beta * nHSIC(spot_embeddings, spot_features)
            HSIC_loss_image = nHSIC(graph_Input, graph_features) - CFG.Beta * HSIC(graph_features, graph_embeddings)
            Alignment_loss = Alignment(graph_embeddings, spot_embeddings)
            loss = Alignment_loss + HSIC_loss_spot + HSIC_loss_image
            loss.backward()
            optimizer.step()
            print('Alignment:', Alignment_loss.item(), '  HSIC_spot:',
                  HSIC_loss_spot.item(), '  HSIC_image:', HSIC_loss_image.item())

    def train_dgi_spot(self, gene_fts, Adj):
        dgi_loss = self.spot_dgi_model(gene_fts.float(), Adj)
        print(' dgi_spot:', dgi_loss)

    def train_dgi_image(self, graph_batch, Adj):
        batch_size = 16;
        for x in range(int((graph_batch.shape[0] + 1) / batch_size) + 1):
            if x == 0:
                graph_fts = self.image_encoder(graph_batch[batch_size * x:batch_size * x + batch_size])
            elif batch_size * x + batch_size - 1 <= graph_batch.shape[0]:
                t = self.image_encoder(graph_batch[batch_size * x:batch_size * x + batch_size])
                graph_fts = torch.cat([graph_fts, t])
            else:
                t = self.image_encoder(graph_batch[batch_size * x:graph_batch.shape[0]])
                graph_fts = torch.cat([graph_fts, t])
        dgi_loss = self.image_dgi_model(graph_fts.float(), Adj)
        print('image dgi:', dgi_loss)


def Alignment(image_embeddings, spot_embeddings):
    logits = (spot_embeddings @ image_embeddings.T) / CFG.temperature
    images_similarity = image_embeddings @ image_embeddings.T
    spots_similarity = spot_embeddings @ spot_embeddings.T
    targets = F.softmax(
        (images_similarity + spots_similarity) / 2 * CFG.temperature, dim=-1
    )
    spots_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)

    return loss.mean()


def HSIC(X, Y):
    N = X.shape[0]
    K_X = torch.pow(X, 2).sum(1, keepdim=True).expand(N, N)
    K_X = torch.clamp(K_X + K_X.T - 2 * X @ X.T, min=0)
    K_X = torch.exp(-K_X / (CFG.sigma ** 2))
    K_Y = torch.pow(Y, 2).sum(1, keepdim=True).expand(N, N)
    K_Y = torch.clamp(K_Y + K_Y.T - 2 * Y @ Y.T, min=0)
    K_Y = torch.exp(-K_Y / (CFG.sigma ** 2))
    H = torch.eye(N) - torch.ones((N, N)) / N
    K_Y = K_Y.cuda()
    K_X = K_X.cuda()
    H = H.cuda()
    HSIC_loss = (torch.trace(K_X @ H @ K_Y @ H)) / ((N - 1) ** 2)
    return HSIC_loss


def nHSIC(X, Y):
    nHSIC_loss = HSIC(X, Y) / (torch.sqrt(1 + HSIC(X, X)) * torch.sqrt(1 + HSIC(Y, Y)))
    return nHSIC_loss
