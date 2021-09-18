import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm
import os
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# - parameters
model_to_test = "models/converted_vissl_swav_covid_e950_e18_layer4_fc.torch"
csv_path = "results.csv"
#model_to_test = sys.argv[1]
label_names = ["COVID", "normal", "pneumonia"]

# transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

compose = transforms.Compose([transforms.Resize((224, 224)),
                              transforms.ToTensor(),
                              normalize])

# init datasets
traindata = ImageFolder("COVIDNet_ImageFolder/train", transform=compose)
testdata = ImageFolder("COVIDNet_ImageFolder/test", transform=compose)

# init dataloader
test_loader = DataLoader(testdata)
# init model and copy weights
network = models.resnet50(pretrained=True)
network.fc = nn.Linear(2048, 3)
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
disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=label_names)
# TODO set colorscheme and fix it to percentages
disp.plot()
fig = disp.figure_
name = model_to_test.split("/")[1].split(".")[0]
fig.savefig(f"plots/{name}.jpg")

label_names = ["COVID", "normal", "pneumonia"]
cf_report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
cf_report
output = []
for label in label_names:
    print(cf_report[label])
    for k, v in cf_report[label].items():
        output.append({"model": name + "y", "label": label, "metric": k, "value": v})
output.append({"model": name + "y", "label": None, "metric": "accuracy", "value": cf_report["accuracy"]})
new_data = pd.DataFrame.from_dict(output)
new_data = new_data.set_index(["model", "label", "metric"])
if os.path.isfile(csv_path):
    df = pd.read_csv(csv_path, index_col=["model", "label", "metric"])
    df = df.combine_first(new_data)
else:
    df = new_data
df.to_csv(csv_path)
