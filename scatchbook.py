from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms, models

from tqdm import tqdm 
import umap 
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import pandas as pd


class Identity(nn.Module):
    """used for skipping last layers"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

def plot_embedding(embedding, labels, outfile_png, outfile_pdf=None):
    """plot a given 2d embedding with right coloring, and save to file"""

    pdframe = pd.DataFrame(
        {"x": embedding[:, 0], "y": embedding[:, 1], "label": list(labels)})

    dim = (12, 12)
    fig = plt.figure(figsize=dim)
    ax = fig.add_subplot(111)

    g = sns.scatterplot(x="x", y="y", hue="label", data=pdframe, palette=sns.color_palette("tab10")[:3], marker="o", s=10, alpha=0.7, edgecolor=None, ax=ax)

    # place legend on the right
    #plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    if outfile_pdf:
        plt.savefig(outfile_pdf, format="pdf")  
    plt.savefig(outfile_png, format="png") 
    plt.close()


# transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

compose = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize])

# init datasets
traindata = ImageFolder("COVIDNet_ImageFolder/train", transform=compose)
testdata = ImageFolder("COVIDNet_ImageFolder/test", transform=compose)

# init dataloader
train_loader = DataLoader(traindata, batch_size=16)
test_loader = DataLoader(testdata)

# init model and copy weights
network = models.resnet50(pretrained=False).to("cuda")
network.load_state_dict(torch.load("models/converted_vissl_swav_covid_e250.torch"), strict=False)
network.fc = Identity()

# loop to obtain features and labels
feats = []
labels = []
for data, label in tqdm(train_loader):
    out = network(data.cuda())
    feats.append(out.detach().cpu())
    labels.append(label)

feats = torch.cat(feats)
feats = feats.view(feats.shape[0], -1).numpy()
labels = torch.cat(labels).numpy()

# perform UMAP to obtain a 2d embedding
embedding = umap.UMAP(verbose=True).fit_transform(feats)

# plot
plot_embedding(embedding=embedding, labels=labels, outfile_png="swav_e250.png")