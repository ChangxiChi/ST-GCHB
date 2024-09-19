import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
from tqdm import tqdm

import os
import cv2
import pandas as pd
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image

num_slice=12


class Crop_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path):
        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)
        self.reduced_matrix = np.load(reduced_mtx_path).T  # cell x features

        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx, 0].split(',')[1]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0]
        # print(v1)
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0]
        image = self.whole_image[(v1 - 56):(v1 + 56), (v2 - 56):(v2 + 56)]
        image = self.transform(image)

        item['image'] = torch.tensor(image).permute(2, 0, 1).float()  # color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx, :]).float()  # cell x features (3467)
        item['barcode'] = barcode
        item['spatial_coords'] = [v1, v2]

        return item

    def __len__(self):
        return len(self.barcode_tsv)


def build_loader(cur):
    # slice 3 randomly chosen to be tested and will be left out during training
    print("Building loaders")
    dataset1 = Crop_Dataset(image_path="image/full_image"+str(cur)+".tif",
                           spatial_pos_path="data/tissue_pos_matrices/tissue_positions_list_"+str(cur)+".csv",
                           reduced_mtx_path="data/filtered_expression_matrices/"+str(cur)+"/harmony"+str(cur)+".npy",
                           barcode_path="data/filtered_expression_matrices/"+str(cur)+"/barcodes"+str(cur)+".tsv")
    # print(adj1.shape)
    # adj1.shape:(4226, 4226)

    return dataset1



for s in range(1,(1+num_slice)):
    cur=s
    train_loader = build_loader(cur)
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    key=0
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
        if key == 0:
            graph_batch = batch["image"]
            graph_batch = graph_batch.unsqueeze(0)
            key = 1
        else:
            graph = batch["image"]
            graph = graph.unsqueeze(0)
            graph_batch = torch.cat([graph_batch, graph])
    torch.save(graph_batch, "data/graph_batch_"+str(cur)+".pt")
    print(graph_batch.shape)
