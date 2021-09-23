from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms, models

from tqdm import tqdm 
import umap 
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
depth = 18
binary = False
model_name = "covid_resnet18_e950_trained_full"
model_path = f"models/{model_name}/model_e90.torch"
#model_path = "vissl_models/converted_vissl_swav_covid_resnet18_e950.torch"
output_name = "train"
save_pdf = True

if binary:
    dataset = "COVIDNet_ImageFolder_binary/train"
    classes = 2
else:
    dataset = "COVIDNet_ImageFolder/train"
    classes = 3

if not os.path.isdir("embeddings"):
    os.makedirs("embeddings")

output_name = "embeddings/" + model_name + output_name + ".png"

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


# - transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

compose = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               normalize])

# - init datasets
traindata = ImageFolder(dataset, transform=compose)

# init dataloader
train_loader = DataLoader(traindata, batch_size=32, shuffle=False)

# init model and copy weights
if depth == 18:
    network = models.resnet18(pretrained=True)
    network.fc = nn.Linear(512, classes)
elif depth == 50:
    network = models.resnet50(pretrained=True)
    network.fc = nn.Linear(2048, classes)

if not model_path == "":
    network.load_state_dict(torch.load(model_path), strict=False)

network.fc = Identity()
network = network.cuda()
network.eval()

# - loop to obtain features and labels
feats = []
labels = []
for data, label in tqdm(train_loader):
    out = network(data.cuda())
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