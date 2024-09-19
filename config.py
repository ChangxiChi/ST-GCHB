
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 4

image_embedding = 1000
spot_embedding =813 #number of shared hvgs (change for each dataset)

pretrained = True
trainable = True
temperature=1
# image size
size = 112

train_dgi_spt_epoch=50
train_dgi_img_epoch=50

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256
hid_dim=768
dropout = 0.4

# make adjacency matrix
topk = 6
radius = 1000

# DGI
Alpha = 0.1
Beta = 25  # in HSIC-bottleneck
test_sampler=3 # have 3 slices to be trained
sigma=10
repeat=800


Num=2;