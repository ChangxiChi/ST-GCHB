import os
import cv2
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image


class ST_GCHB_Dataset(torch.utils.data.Dataset):
    def __init__(self,  spatial_pos_path, barcode_path, reduced_mtx_path):

        # self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header=None)
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header=None)
        self.reduced_matrix = np.load(reduced_mtx_path).T  # cell x features

        print("Finished loading all files")


    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx, 0].split(',')[1]

        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 4].values[0]
        # print(v1)
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode, 5].values[0]

        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx, :]).float()  # cell x features (3467)
        item['barcode'] = barcode
        item['spatial_coords'] = [v1, v2]

        return item

    def __len__(self):
        return len(self.barcode_tsv)