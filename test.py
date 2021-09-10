import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import seaborn as sns

# - parameters
model_to_test = "models/train_full/converted_vissl_swav_covid_e950_e90.torch"


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
network = models.resnet50(pretrained=False)
network.fc = nn.Linear(2048, 3)
network.load_state_dict(torch.load(model_to_test))
network.eval()
network = network.cuda()

total = 0
correct = 0
with torch.no_grad():
    for data, label in tqdm(test_loader):
        data = data.cuda()
        label = label.cuda()
        out = network(data)
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print(f'Accuracy:{correct/total}')
