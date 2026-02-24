import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_ROOT = "data/imagenette2-160"

# 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Normal taber-cuda eller sindssyg Mac M-chip:
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using GPU.")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")
    print("MPS backend is not available. Using CPU.")

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_tfms)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=64, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model.to(device)

# freeze backbone
for name, p in model.named_parameters():
    if not name.startswith("fc."):
        p.requires_grad = False

opt = torch.optim.AdamW(model.fc.parameters(), lr=3e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for x,y in train_dl:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    print("epoch", epoch+1, "done")