import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm
import os
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# - parameters
model_to_test = "models/binary_resnet18_ImageNet_trained_full_20val/model_e11_best.torch"
csv_path = "results.csv"
model_to_test = sys.argv[1]

# transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if "480" in model_to_test:
    compose = transforms.Compose([transforms.Resize((480, 480)),
                                  transforms.ToTensor(),
                                  normalize])
else:
    compose = transforms.Compose([transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  normalize])



if "binary" in model_to_test:
    # init datasets
    traindata = ImageFolder("COVIDNet_ImageFolder_bin/train", transform=compose)
    testdata = ImageFolder("COVIDNet_ImageFolder_bin/test", transform=compose)

    # init dataloader
    test_loader = DataLoader(testdata)
    label_names = ["COVID", "other"]
    out_size = 2
else:
    # init datasets
    traindata = ImageFolder("COVIDNet_ImageFolder/train", transform=compose)
    testdata = ImageFolder("COVIDNet_ImageFolder/test", transform=compose)

    # init dataloader
    test_loader = DataLoader(testdata)
    label_names = ["COVID", "normal", "pneumonia"]
    out_size = 3
# init model and copy weights
if "resnet18" in model_to_test:
    network = models.resnet18(pretrained=True)
    network.fc = nn.Linear(512, out_size)
else:
    network = models.resnet50(pretrained=True)
    network.fc = nn.Linear(2048, out_size)


network.load_state_dict(torch.load(model_to_test))
network.eval()
network = network.cuda()

total = 0
correct = 0

y_pred = []
y_true = []
with torch.no_grad():
    for data, label in tqdm(test_loader):
        data = data.cuda()
        label = label.cuda()
        out = network(data)
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        y_true.extend(label.cpu())
        y_pred.extend(predicted.cpu())

    print(f'Accuracy:{correct / total}')

print(classification_report(y_true, y_pred, target_names=label_names))
cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
ax = sns.heatmap(cf_matrix, annot=True, yticklabels=label_names, xticklabels=label_names, cbar=False, annot_kws={"size": 15})
ax.set_aspect('equal', 'box')
plt.xlabel("Predicted label")
plt.ylabel("True label")
name = model_to_test.split("/")[1].split(".")[0]
name = "_".join(model_to_test.split("/")[1:3]).split(".")[0]
plt.savefig(f"plots/{name}.pdf")


cf_report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
output = []
for label in label_names:
    print(cf_report[label])
    for k, v in cf_report[label].items():
        output.append({"model": name, "label": label, "metric": k, "value": v})
output.append({"model": name, "label": None, "metric": "accuracy", "value": cf_report["accuracy"]})
new_data = pd.DataFrame.from_dict(output)
new_data = new_data.set_index(["model", "label", "metric"])
if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path, index_col=["model", "label", "metric"])
    df = df.combine_first(new_data)
else:
    df = new_data
df.to_csv(csv_path)
