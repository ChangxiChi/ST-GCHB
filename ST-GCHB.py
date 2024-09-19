import os
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch.nn.functional as F
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import config as CFG
from dataset import ST_GCHB_Dataset
from models import cross_entropy,ST_GCHB
from sklearn import neighbors
import pandas as pd
import scipy.sparse
from modules import dim_reduce


traing_slice_num=10

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def Cal_Spatial_Net(dataset, rad=CFG.radius, k=CFG.topk):  # X表示每个spot的坐标数据
    X = torch.Tensor()  # tensor([])
    for idx in range(len(dataset)):
        if idx == 0:
            X = np.append(X, dataset[idx]['spatial_coords'])  # tensor([a,b])
            X = torch.from_numpy(X)  # from array to tensor
            X = X.unsqueeze(0)  # shape:[[a,b]]
        else:
            coor = [dataset[idx]['spatial_coords']]
            X = np.append(X, coor, axis=0)
    # print("X type is:", type(X))
    nbrs = neighbors.NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    # print(KNN_list)
    adj = np.zeros((indices.shape[0], indices.shape[0]))
    KNN_list = np.array(KNN_list)
    for idx in range(len(KNN_list)):
        t = KNN_list[idx, :]
        for p in range(len(t)):
            x = int(t[p, 0])
            y = int(t[p, 1])
            adj[x, y] = 1
    return adj


# select your traing slices
def build_loaders():
    print("Building loaders")
    dataset1 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_1.csv",
        reduced_mtx_path="data/filtered_expression_matrices/1/harmony1.npy",
        barcode_path="data/filtered_expression_matrices/1/barcodes1.tsv")
    adj1 = Cal_Spatial_Net(dataset1, CFG.radius, CFG.topk)
    adj1 = scipy.sparse.coo_matrix(adj1)

    dataset2 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_2.csv",
        reduced_mtx_path="data/filtered_expression_matrices/2/harmony2.npy",
        barcode_path="data/filtered_expression_matrices/2/barcodes2.tsv")
    adj2 = Cal_Spatial_Net(dataset2, CFG.radius, CFG.topk)
    adj2 = scipy.sparse.coo_matrix(adj2)

    dataset3 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_3.csv",
        reduced_mtx_path="data/filtered_expression_matrices/3/harmony3.npy",
        barcode_path="data/filtered_expression_matrices/3/barcodes3.tsv")
    adj3 = Cal_Spatial_Net(dataset3, CFG.radius, CFG.topk)
    adj3 = scipy.sparse.coo_matrix(adj3)

    dataset4 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_4.csv",
        reduced_mtx_path="data/filtered_expression_matrices/4/harmony4.npy",
        barcode_path="data/filtered_expression_matrices/4/barcodes4.tsv")
    adj4 = Cal_Spatial_Net(dataset4, CFG.radius, CFG.topk)
    adj4 = scipy.sparse.coo_matrix(adj4)

    dataset5 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_5.csv",
        reduced_mtx_path="data/filtered_expression_matrices/5/harmony5.npy",
        barcode_path="data/filtered_expression_matrices/5/barcodes5.tsv")
    adj5 = Cal_Spatial_Net(dataset5, CFG.radius, CFG.topk)
    adj5 = scipy.sparse.coo_matrix(adj5)

    dataset6 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_6.csv",
        reduced_mtx_path="data/filtered_expression_matrices/6/harmony6.npy",
        barcode_path="data/filtered_expression_matrices/6/barcodes6.tsv")
    adj6 = Cal_Spatial_Net(dataset6, CFG.radius, CFG.topk)
    adj6 = scipy.sparse.coo_matrix(adj6)

    dataset7 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_7.csv",
        reduced_mtx_path="data/filtered_expression_matrices/7/harmony7.npy",
        barcode_path="data/filtered_expression_matrices/7/barcodes7.tsv")
    adj7 = Cal_Spatial_Net(dataset7, CFG.radius, CFG.topk)
    adj7 = scipy.sparse.coo_matrix(adj7)

    dataset8 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_8.csv",
        reduced_mtx_path="data/filtered_expression_matrices/8/harmony8.npy",
        barcode_path="data/filtered_expression_matrices/8/barcodes8.tsv")
    adj8 = Cal_Spatial_Net(dataset8, CFG.radius, CFG.topk)
    adj8 = scipy.sparse.coo_matrix(adj8)

    dataset9 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_9.csv",
        reduced_mtx_path="data/filtered_expression_matrices/9/harmony9.npy",
        barcode_path="data/filtered_expression_matrices/9/barcodes9.tsv")
    adj9 = Cal_Spatial_Net(dataset9, CFG.radius, CFG.topk)
    adj9 = scipy.sparse.coo_matrix(adj9)

    dataset10 = ST_GCHB_Dataset(
        spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_10.csv",
        reduced_mtx_path="data/filtered_expression_matrices/10/harmony10.npy",
        barcode_path="data/filtered_expression_matrices/10/barcodes10.tsv")
    adj10 = Cal_Spatial_Net(dataset10, CFG.radius, CFG.topk)
    adj10 = scipy.sparse.coo_matrix(adj10)

    return (dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8,dataset9,dataset10,
            adj1, adj2, adj3, adj4, adj5, adj6, adj7, adj8, adj9, adj10)



A_dict={}
Image_dict={}

(train_loader1, train_loader2, train_loader3, train_loader4, train_loader5, train_loader6, train_loader7,
 train_loader8, train_loader9, train_loader10,
 Adj1, Adj2, Adj3, Adj4, Adj5, Adj6, Adj7, Adj8, Adj9, Adj10) = build_loaders()
A_dict[0] = sparse_mx_to_torch_sparse_tensor(Adj1).cuda()
A_dict[1] = sparse_mx_to_torch_sparse_tensor(Adj2).cuda()
A_dict[2] = sparse_mx_to_torch_sparse_tensor(Adj3).cuda()
A_dict[3] = sparse_mx_to_torch_sparse_tensor(Adj4).cuda()
A_dict[4] = sparse_mx_to_torch_sparse_tensor(Adj5).cuda()
A_dict[5] = sparse_mx_to_torch_sparse_tensor(Adj6).cuda()
A_dict[6] = sparse_mx_to_torch_sparse_tensor(Adj7).cuda()
A_dict[7] = sparse_mx_to_torch_sparse_tensor(Adj8).cuda()
A_dict[8] = sparse_mx_to_torch_sparse_tensor(Adj9).cuda()
A_dict[9] = sparse_mx_to_torch_sparse_tensor(Adj10).cuda()




Image_dict[0] = torch.load('data/graph_batch_1.pt').to('cpu')
Image_dict[1] = torch.load('data/graph_batch_2.pt').to('cpu')
Image_dict[2] = torch.load('data/graph_batch_3.pt').to('cpu')
Image_dict[3] = torch.load('data/graph_batch_4.pt').to('cpu')
Image_dict[4] = torch.load('data/graph_batch_5.pt').to('cpu')
Image_dict[5] = torch.load('data/graph_batch_6.pt').to('cpu')
Image_dict[6] = torch.load('data/graph_batch_7.pt').to('cpu')
Image_dict[7] = torch.load('data/graph_batch_8.pt').to('cpu')
Image_dict[8] = torch.load('data/graph_batch_9.pt').to('cpu')
Image_dict[9] = torch.load('data/graph_batch_10.pt').to('cpu')

Gene_fts_1 = torch.tensor(train_loader1.reduced_matrix).cuda()
Gene_fts_2 = torch.tensor(train_loader2.reduced_matrix).cuda()
Gene_fts_3 = torch.tensor(train_loader3.reduced_matrix).cuda()
Gene_fts_4 = torch.tensor(train_loader4.reduced_matrix).cuda()
Gene_fts_5 = torch.tensor(train_loader5.reduced_matrix).cuda()
Gene_fts_6 = torch.tensor(train_loader6.reduced_matrix).cuda()
Gene_fts_7 = torch.tensor(train_loader7.reduced_matrix).cuda()
Gene_fts_8 = torch.tensor(train_loader8.reduced_matrix).cuda()
Gene_fts_9 = torch.tensor(train_loader9.reduced_matrix).cuda()
Gene_fts_10 = torch.tensor(train_loader10.reduced_matrix).cuda()


List = [Gene_fts_1.shape[0], Gene_fts_2.shape[0], Gene_fts_3.shape[0], Gene_fts_4.shape[0], Gene_fts_5.shape[0],
        Gene_fts_6.shape[0], Gene_fts_7.shape[0], Gene_fts_8.shape[0], Gene_fts_9.shape[0], Gene_fts_10.shape[0]]
data = torch.cat([Gene_fts_1, Gene_fts_2, Gene_fts_3, Gene_fts_4, Gene_fts_5, Gene_fts_6, Gene_fts_7,
                  Gene_fts_8, Gene_fts_9, Gene_fts_10])

# (Gene_fts_1,Gene_fts_2,Gene_fts_3,Gene_fts_4,Gene_fts_5,Gene_fts_6,
#  Gene_fts_7,Gene_fts_8,Gene_fts_9,Gene_fts_10) = dim_reduce(List,data)

gene_fts_all = dim_reduce(List,data)

ST_Dict={i:{'Gene_fts':gene_fts_all[i],'Adj':A_dict[i],'Graph':Image_dict[i]} for i in range(traing_slice_num)}

def train_epoch(model, optimizer):
    torch.cuda.empty_cache()
    for slice in range(traing_slice_num):
        graph_batch = ST_Dict[slice]['Graph']
        gene_fts = ST_Dict[slice]['Gene_fts']
        adj = ST_Dict[slice]['Adj']

        model(graph_batch,gene_fts,adj,optimizer)
        torch.cuda.empty_cache()


def train_dgi_spot(model):
    for epoch in range(CFG.train_dgi_spt_epoch):
        for i in range(traing_slice_num):
            gene_fts = ST_Dict[i]['Gene_fts']
            Adj = ST_Dict[slice]['Adj']

            model.train_dgi_spot(gene_fts,Adj)


def train_dgi_image(model):
    for epoch in range(CFG.train_dgi_img_epoch):
        for i in range(traing_slice_num):
            graph_batch = ST_Dict[i]['Graph']
            Adj = ST_Dict[i]['Adj']

            model.train_dgi_image(graph_batch,Adj)



def main():
    print("Starting...")
    print("process group ready!")

    model=ST_GCHB().cuda()
    model.train()
    para_gruops = [{'params': model.parameters(), 'lr': CFG.lr}]

    optimizer = torch.optim.AdamW(para_gruops, lr=0.01, weight_decay=CFG.weight_decay)
    train_dgi_image(model)
    train_dgi_spot(model)
    torch.cuda.empty_cache()
    for i in range(CFG.epochs):
        print("epoch: ",i)
        train_epoch(model, optimizer)
        torch.cuda.empty_cache()
        torch.save(model.state_dict(),"./ST_GCHB.pt")


if __name__ == "__main__":
    main()
