from src.evaluate import evaluate, compute_metrics
from typing import Dict, List
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path(".").resolve()
ARTIFACTS_DIR = ROOT / "artifacts"

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            return float("nan"), float("nan")

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
        total_seen += bs

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def plot_confusion_matrix(hist):
    descr = f"Epoch: {hist['epoch']}"
    cm = hist['cm']
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    label_names = ["NORMAL", "PNEUMONIA"]
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(descr)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png")
    plt.show()


def fit(model, train_loader, val_loader, optimizer, criterion,
        epochs=10, verbose=True):
    print("Trainable params:", count_params(model))

    history = []
    best_f1 = -1.0
    best_epoch_metrics = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # train
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        # val
        va_loss, y_true, y_pred, y_prob = evaluate(
            model,
            val_loader,
            criterion
        )

        if y_true is None:
            print("NaN/Inf during validation")
            break

        metrics = compute_metrics(y_true, y_pred, y_prob)

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "y_true": y_true,
            **metrics
        }

        history.append(row)

        # save best f1-score
        if metrics["f1"] > best_f1:
            best_epoch_metrics = row
            best_f1 = metrics["f1"]

        dt = time.time() - t0

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f}, "
                f"acc {metrics['acc']:.4f}, "
                f"recall {metrics['recall']:.4f}, "
                f"f1 {metrics['f1']:.4f}, "
                f"auc {metrics['roc_auc']:.4f} | "
                f"{dt:.1f}s"
            )

        if (not math.isfinite(tr_loss)) or (not math.isfinite(va_loss)):
            print("NaN/Inf detected. Stop training.")
            break

    # вывод confusion matrix лучшей эпохи обучения
    plot_confusion_matrix(best_epoch_metrics)

    history = pd.DataFrame(history)

    return history


def plot_history(hist: Dict[str, List[float]], title: str = "", save_name: str = None) -> None:
    epochs = list(range(1, len(hist["train_loss"]) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, hist["train_loss"], label="train loss")
    plt.plot(epochs, hist["train_acc"], label="train acc")
    plt.xlabel("epoch")
    # plt.ylabel("loss")
    plt.title(title + " | train")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, hist["val_loss"], label="val loss")
    plt.plot(epochs, hist["acc"], label="val acc")
    plt.plot(epochs, hist["recall"], label="recall")
    plt.xlabel("epoch")
    # plt.ylabel("accuracy")
    plt.title(title + " | val")
    plt.grid(True)
    plt.legend()
    if save_name:
        plt.savefig(ARTIFACTS_DIR / save_name)
    plt.show()


from src.data import load_datasets, get_transforms, make_loaders_model
from src.model import build_resnet18, get_resnet18_weights

if __name__ == "__main__":
    train_tf, eval_tf = get_transforms()
    train_ds, val_ds, test_ds = load_datasets()
    class_names = train_ds.classes
    num_classes = len(class_names)

    train_loader_resnet, val_loader_resnet, test_loader_resnet = make_loaders_model(
        train_ds, val_ds, test_ds, batch_size=64,
    )

    weights = get_resnet18_weights()
    model = build_resnet18(num_classes=num_classes, weights=weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    params = [
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ]
    optimizer_ft = torch.optim.AdamW(params, weight_decay=1e-4)

    history = fit(model, train_loader_resnet, test_loader_resnet, optimizer_ft, criterion, epochs=5, verbose=True)
    plot_history(history, title="ResNet18 | pretrained", save_name="train_model_epochs")
    torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pth")