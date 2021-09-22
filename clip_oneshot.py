import clip
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
import torch
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, recall_score, precision_score

# - parameters
data_set = "COVIDNet_ImageFolder_binary"
#data_set = "chexpert_covid_ImageFolder"
#data_set = "COVIDNet_ImageFolder"
device = "cuda"

# - init
#model, preprocess = clip.load('RN50', device = device)
model, preprocess = clip.load('ViT-B/32', device = device)
dataset = ImageFolder(f"{data_set}/test")

# Calculate text features
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c} lung") for c in dataset.classes]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

all_pred = []
# - loop 
for image, label in tqdm(dataset):
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Calculate image features
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Pick the most similar label for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)

    all_pred.append(dataset.classes[indices])
    #for value, index in zip(values, indices):
    #    print(f"{dataset.classes[index]:>16s}: {100 * value.item():.2f}%")

reverse_labels = {v:k for k,v in dataset.class_to_idx.items()}
labels = [reverse_labels.get(item, item) for item in dataset.targets]

# in the micro average accuracy, recall and precision are the same
acc = accuracy_score(labels, all_pred)
rec = recall_score(labels, all_pred, average="micro")
pre = precision_score(labels, all_pred, average="micro")

print(acc)
print(rec)
print(pre)
