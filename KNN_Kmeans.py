import faiss
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from collections import Counter

from tqdm import tqdm 
import umap 
import clip
import faiss
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# - to overwrite classification layer
class Identity(nn.Module):
    """used for skipping last layers"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# - parameters 
binary = False
use_clip = True

use_umap = False
use_kmeans = False

# set model path for pre-trained models; if "" - use pre-trained ImageNet; only for not-clip
#model_path = "vissl_models/converted_vissl_swav_covid_e950.torch"
model_path = ""

if binary:
    traindataset = "COVIDNet_ImageFolder_binary/train"
    testdataset = "COVIDNet_ImageFolder_binary/test"
    classes = 2
    label_names = ["COVID-19", "healthy"]
else:
    traindataset = "COVIDNet_ImageFolder/train"
    testdataset = "COVIDNet_ImageFolder/test"
    classes = 3
    label_names = ["COVID-19", "normal", "pneumonia"]

# - init model and transform 
if use_clip:
    print("using clip model")
    model, preprocess = clip.load('ViT-B/32', device="cuda")
else:
    print("using resnet50 model")
    model = models.resnet50(pretrained=True)

    if not model_path == "":
        print(f"loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path), strict=False)

    model.fc = Identity()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               normalize])
    model = model.cuda()
    model.eval()


# - init datasets
traindata = ImageFolder(traindataset, transform=preprocess)
testdata = ImageFolder(testdataset, transform=preprocess)

# init dataloader
trainloader = DataLoader(traindata, batch_size=64, shuffle=False)
testloader = DataLoader(testdata, batch_size=64, shuffle=False)

trainfeats = []
trainlabels = []
for data, label in tqdm(trainloader):

    if use_clip:    
        with torch.no_grad():
            out = model.encode_image(data)
    else:
        out = model(data.cuda())

    trainfeats.append(out.detach().cpu())
    trainlabels.append(label)


testfeats = []
testlabels = []
for data, label in tqdm(testloader):

    if use_clip:    
        with torch.no_grad():
            out = model.encode_image(data)
    else:
        out = model(data.cuda())

    testfeats.append(out.detach().cpu())
    testlabels.append(label)

testfeats = torch.cat(testfeats)
testfeats = testfeats.view(testfeats.shape[0], -1).numpy()

testlabels = torch.cat(testlabels).numpy()
testreverse_labels = {v:k for k,v in testdata.class_to_idx.items()}
testlabels = [testreverse_labels.get(item, item) for item in testlabels]

# - flatten data
trainfeats = torch.cat(trainfeats)
trainfeats = trainfeats.view(trainfeats.shape[0], -1).numpy()

trainlabels = torch.cat(trainlabels).numpy()
reverse_labels = {v:k for k,v in traindata.class_to_idx.items()}
trainlabels = [reverse_labels.get(item, item) for item in trainlabels]

if use_umap:
    embedding = umap.UMAP(verbose=True).fit_transform(testfeats)
    testfeats = embedding

if use_kmeans:
    kmeans = faiss.Kmeans(d=testfeats.shape[1], k=classes)
    kmeans.train(testfeats.astype(np.float32))

    _, c_labels = kmeans.index.search(testfeats.astype(np.float32), 1)
    c_labels = c_labels.flatten()

    clustering = {cl: np.where(c_labels == cl)[0] for cl in np.unique(c_labels)}


    pred_labels = np.zeros_like(testlabels)
    for c_idcs in clustering.values():

        gt = np.array(testlabels)
        gt_labels = gt[c_idcs]
        label_counter = Counter(gt_labels)
        pred_lbl = list(label_counter.keys())[np.argmax(list(label_counter.values()))]
        pred_labels[c_idcs] = pred_lbl

else:
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(trainfeats, trainlabels)
    pred_labels = neigh.predict(testfeats)


#print(labels)
#print(pred_labels)
#exit()

print(classification_report(testlabels, pred_labels, target_names=label_names))


