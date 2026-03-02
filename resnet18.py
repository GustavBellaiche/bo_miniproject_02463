import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


# -----------------------------
# Device (Mac MPS / CUDA / CPU)
# -----------------------------
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print("Device:", DEVICE)

# -----------------------------
# Data config
# -----------------------------
DATA_ROOT = "data/imagenette2-160"
BATCH_SIZE = 64
NUM_WORKERS = 2
IMG_SIZE = 224

# -----------------------------
# BO config (head-only)
# -----------------------------
SEED = 42
EPOCHS_PER_TRIAL = 1

N_INIT = 8
N_TRIALS = 30
NUM_RESTARTS = 10
RAW_SAMPLES = 256

# Search space (unit cube -> decode)
# lr_head: log-uniform [1e-4, 3e-2]
# weight_decay: log-uniform [1e-6, 1e-2]
# label_smoothing: uniform [0.0, 0.2]
BOUNDS = {
    "lr_head": (1e-4, 3e-2),
    "weight_decay": (1e-6, 1e-2),
    "label_smoothing": (0.0, 0.2),
}
D = 3


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(batch_size: int):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_tfms)

    pin = (DEVICE == "cuda")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    return train_ds, val_ds, train_dl, val_dl


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone_forever(model: nn.Module):
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = False


def make_optimizer_head_only(model: nn.Module, lr_head: float, weight_decay: float):
    return torch.optim.AdamW(model.fc.parameters(), lr=lr_head, weight_decay=weight_decay)


def log_uniform(u: float, lo: float, hi: float) -> float:
    return float(10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo))))


def lin_uniform(u: float, lo: float, hi: float) -> float:
    return float(lo + u * (hi - lo))


def decode_params(X_unit: torch.Tensor):
    """
    X_unit: (n,3) in [0,1]
    returns list of dict hyperparams
    """
    X = X_unit.detach().cpu().numpy()
    out = []
    for u_lr, u_wd, u_ls in X:
        out.append({
            "lr_head": log_uniform(u_lr, *BOUNDS["lr_head"]),
            "weight_decay": log_uniform(u_wd, *BOUNDS["weight_decay"]),
            "label_smoothing": lin_uniform(u_ls, *BOUNDS["label_smoothing"]),
        })
    return out


@torch.no_grad()
def evaluate(model, val_dl, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for x, y in val_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)

    return total_loss / n, correct / n


def train_one_epoch(model, train_dl, optimizer, criterion):
    model.train()
    total_loss, n = 0.0, 0

    for x, y in tqdm(train_dl, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / n


def objective(train_dl, val_dl, num_classes: int, hparams: dict, seed: int):
    """
    BO maksimerer objective => vi maksimerer -val_loss.
    Return: (obj, val_loss, val_acc)
    """
    seed_everything(seed)

    model = build_model(num_classes).to(DEVICE)
    freeze_backbone_forever(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=hparams["label_smoothing"])
    optimizer = make_optimizer_head_only(model, lr_head=hparams["lr_head"], weight_decay=hparams["weight_decay"])

    # 1 epoch budget
    train_one_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_dl, criterion)

    obj = -val_loss
    return float(obj), float(val_loss), float(val_acc)


def bo_loop(train_dl, val_dl, num_classes: int):
    logs = []
    t0 = time.time()

    # Sobol init in unit cube
    X = draw_sobol_samples(
        bounds=torch.stack([torch.zeros(D), torch.ones(D)]),
        n=N_INIT,
        q=1,
    ).squeeze(1)  # (N_INIT, D)

    Y_list = []

    # ---- eval init points ----
    for i in range(N_INIT):
        hp = decode_params(X[i:i+1])[0]
        obj, vloss, vacc = objective(train_dl, val_dl, num_classes, hp, seed=SEED + i)

        Y_list.append([obj])
        logs.append({"iter": i, **hp, "objective": obj, "val_loss": vloss, "val_acc": vacc})

        print(f"[init {i+1}/{N_INIT}] val_loss={vloss:.4f} val_acc={vacc:.4f} hp={hp}")

    Y = torch.tensor(Y_list, dtype=torch.double)  # (n,1)

    # ---- BO iterations ----
    for it in range(N_INIT, N_TRIALS):
        train_X = X.to(dtype=torch.double)
        train_Y = Y

        gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        best_f = train_Y.max()
        acq = qLogExpectedImprovement(model=gp, best_f=best_f)

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(D, dtype=torch.double), torch.ones(D, dtype=torch.double)]),
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
        )

        x_next = candidate.detach().clamp(0.0, 1.0)  # (1,D)
        hp = decode_params(x_next)[0]

        obj, vloss, vacc = objective(train_dl, val_dl, num_classes, hp, seed=SEED + it)

        X = torch.cat([X, x_next.to(dtype=X.dtype)], dim=0)
        Y = torch.cat([Y, torch.tensor([[obj]], dtype=torch.double)], dim=0)

        best_loss_so_far = float((-Y).min())  # since obj = -loss
        logs.append({"iter": it, **hp, "objective": obj, "val_loss": vloss, "val_acc": vacc})

        print(f"[bo {it+1}/{N_TRIALS}] val_loss={vloss:.4f} val_acc={vacc:.4f} best_loss={best_loss_so_far:.4f} hp={hp}")

    best_idx = int(Y.argmax().item())
    best_hp = decode_params(X[best_idx:best_idx+1])[0]
    best_obj = float(Y[best_idx].item())

    print("\nBest found:")
    print("  hp:", best_hp)
    print("  best val_loss:", -best_obj)
    print("Time (s):", round(time.time() - t0, 1))

    return best_hp, logs


def main():
    seed_everything(SEED)

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT findes ikke: {DATA_ROOT}")

    train_ds, val_ds, train_dl, val_dl = make_dataloaders(BATCH_SIZE)
    num_classes = len(train_ds.classes)

    print("Train size:", len(train_ds), "| Val size:", len(val_ds), "| Classes:", num_classes)

    best_hp, logs = bo_loop(train_dl, val_dl, num_classes)

    # gem logs som csv
    import pandas as pd
    pd.DataFrame(logs).to_csv("botorch_headonly_runs.csv", index=False)
    print("Saved: botorch_headonly_runs.csv")


if __name__ == "__main__":
    main()