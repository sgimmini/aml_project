from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms, models

from tqdm import tqdm 
import umap 
import clip
import faiss
import torch
import os
import numpy as np
import torch.nn as nn

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import pandas as pd

font = {'family' : 'serif',
        'size'   : 20}

mpl.rc('font', **font)
mpl.rc("xtick", labelsize=20)
mpl.rc("ytick", labelsize=20)

# - parameters
binary = True
output_name = "_deleteme"
save_pdf = True

if binary:
    dataset = "COVIDNet_ImageFolder_binary/test"
    classes = 2
else:
    dataset = "COVIDNet_ImageFolder/test"
    classes = 3

if not os.path.isdir("embeddings"):
    os.makedirs("embeddings")

output_name = "embeddings/" + "clip_vit" + output_name + ".png"

# - to overwrite classification layer
class Identity(nn.Module):
    """used for skipping last layers"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# - plot function
def plot_embedding(embedding, labels, outfile_png, outfile_pdf=None):
    """plot a given 2d embedding with right coloring, and save to file"""

    pdframe = pd.DataFrame(
        {"x": embedding[:, 0], "y": embedding[:, 1], "label": list(labels)})

    dim = (12, 12)
    fig = plt.figure(figsize=dim)
    ax = fig.add_subplot(111)

    g = sns.scatterplot(x="x", 
        y="y", 
        hue="label", 
        data=pdframe, 
        palette=sns.color_palette("tab10")[:len(np.unique(labels))], 
        marker="o", 
        s=30, 
        alpha=0.7, 
        edgecolor=None, 
        ax=ax)

    # place legend on the right
    #plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    if outfile_pdf:
        plt.savefig(outfile_pdf, format="pdf")  
    plt.savefig(outfile_png, format="png") 
    plt.close()


model, preprocess = clip.load('ViT-B/32', device = "cuda")

# - init datasets
traindata = ImageFolder(dataset, transform=preprocess)

# init dataloader
train_loader = DataLoader(traindata, batch_size=32, shuffle=False)


# - loop to obtain features and labels
feats = []
labels = []
for data, label in tqdm(train_loader):
    
    with torch.no_grad():
        out = model.encode_image(data)

    feats.append(out.detach().cpu())
    labels.append(label)

feats = torch.cat(feats)
feats = feats.view(feats.shape[0], -1).numpy()

labels = torch.cat(labels).numpy()
reverse_labels = {v:k for k,v in traindata.class_to_idx.items()}
labels = [reverse_labels.get(item, item) for item in labels]

# - perform UMAP to obtain a 2d embedding
embedding = umap.UMAP(verbose=True).fit_transform(feats)

# - plot
if save_pdf:
    plot_embedding(embedding=embedding, 
        labels=labels, 
        outfile_png=output_name,
        outfile_pdf=output_name.replace("png", "pdf"))
else:
    plot_embedding(embedding=embedding, 
        labels=labels, 
        outfile_png=output_name)