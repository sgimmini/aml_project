from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

from torch.utils.tensorboard import SummaryWriter

# - parameters
NUM_EPOCHS = 100
depth = 50
use_libauc = False
val_size = 0.2

experiment_name = "binary_deepcluster_e950_trained_full_20val"
pretrained_model = "converted_vissl_deepcluster_covid_e950.torch"
pretrained_model_path = f"vissl_models/{pretrained_model}"

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
writer = SummaryWriter(log_dir=f"runs/{current_time}_{experiment_name}")

# transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

compose = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               normalize])

# init datasets
traindata = ImageFolder("COVIDNet_ImageFolder_binary/train", transform=compose)

train_len = int((1.0-val_size)*len(traindata))
traindata, validdata = torch.utils.data.random_split(traindata, [train_len, len(traindata)-train_len], generator=torch.Generator().manual_seed(42))

# init dataloader
train_loader = DataLoader(traindata, batch_size=32, shuffle=True)
valid_loader = DataLoader(validdata, batch_size=32, shuffle=True)

# init model and copy weights
if depth == 18:
    network = models.resnet18(pretrained=True)
    network.fc = nn.Linear(512, 2)
if depth == 50:
    network = models.resnet50(pretrained=True)
    network.fc = nn.Linear(2048, 2)

network.load_state_dict(torch.load(pretrained_model_path), strict=False)
network.fc.weight.data.normal_(mean=0.0, std=0.01)
network.fc.bias.data.zero_()

# for name, parameter in network.named_parameters():
#     if any(layer in name for layer in ['fc']):
#         parameter.requires_grad = True
#         print(name)
#     else:
#         parameter.requires_grad = False

network = network.cuda()
network.train()
print(network)

if use_libauc:
    criterion = AUCMLoss()
    optimizer = PESG(network, a=criterion.a, b=criterion.b, alpha=criterion.alpha, lr=0.0002)
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0002)


best_valid_loss = 1000
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    network.train()
    train_loss = 0
    train_total = 0
    train_correct = 0
    for data, label in tqdm(train_loader, desc="Train", leave=False):
        data = data.cuda()
        label = label.cuda()
        out = network(data)
        loss = criterion(out, label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(out.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()
    print('Train ===============================')
    print(f'Accuracy:{train_correct/train_total}')
    writer.add_scalar("Accuracy/train", train_correct/train_total, epoch)
    print(f"epoch: {epoch}, Train Loss: {train_loss/len(train_loader)}")
    writer.add_scalar("Loss/train", train_loss/len(train_loader), epoch)

    valid_loss = 0
    valid_total = 0
    valid_correct = 0
    for data, label in tqdm(valid_loader, desc="Valid", leave=False):
        network.eval()
        data = data.cuda()
        label = label.cuda()
        out = network(data)
        loss = criterion(out, label)
        valid_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        valid_total += label.size(0)
        valid_correct += (predicted == label).sum().item()
    print('Validation ==========================')
    print(f'Accuracy:{valid_correct/valid_total}')
    writer.add_scalar("Accuracy/valid", valid_correct/valid_total, epoch)
    print(f"epoch: {epoch}, Validation Loss: {valid_loss/len(valid_loader)}")
    writer.add_scalar("Loss/valid", valid_loss/len(valid_loader), epoch)

    # - saving
    save_path = f"models/{experiment_name}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # - save best epoch
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        previous_best = f"{save_path}/model_e{best_epoch}_best.torch"
        if os.path.exists(previous_best):
            os.remove(previous_best)
        best_epoch = epoch
        torch.save(network.state_dict(), f"{save_path}/model_e{epoch}_best.torch")

    # save only every 10th epoch
    if epoch % 10 == 0:
        torch.save(network.state_dict(), f"{save_path}/model_e{epoch}.torch")
