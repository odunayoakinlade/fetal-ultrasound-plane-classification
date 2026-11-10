"""Model training and evaluation.

The images come from the Phase 2 pipeline (224 x 224, uint8). Augmentation is applied to
training images only; validation and test images are only scaled for the backbone.
"""

import copy

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from src.transformers import augment, to_model_input


def get_device():
    """Apple GPU if available, then NVIDIA GPU, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PlaneDataset(Dataset):
    """Standardised images and their plane labels, with augmentation for training only."""

    def __init__(self, images, labels, train=False, seed=None):
        self.images = images
        self.labels = np.asarray(labels)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        if self.train:
            image = augment(image, self.rng)
        x = torch.tensor(to_model_input(image), dtype=torch.float32)
        return x, int(self.labels[i])


def make_loader(images, labels, train=False, batch_size=32, seed=None):
    dataset = PlaneDataset(images, labels, train=train, seed=seed)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, generator=generator)


def build_model(num_classes=4):
    """ResNet18 pretrained on ImageNet, with a new final layer for the four planes."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training data; returns the average loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """Returns the true labels, predicted labels and class probabilities."""
    model.eval()
    true, probabilities = [], []
    for x, y in loader:
        probabilities.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        true.append(y.numpy())
    probabilities = np.concatenate(probabilities)
    return np.concatenate(true), probabilities.argmax(axis=1), probabilities


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=10, patience=3):
    """Trains for up to `epochs`, keeping the weights with the best validation macro-F1.

    Stops early if validation macro-F1 has not improved for `patience` epochs.
    Returns the training history (one row per epoch).
    """
    history, best_f1, best_weights, waited = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        true, pred, _ = predict(model, val_loader, device)
        val_f1 = f1_score(true, pred, average="macro")
        val_accuracy = (true == pred).mean()
        history.append({"epoch": epoch, "train loss": train_loss,
                        "validation accuracy": val_accuracy, "validation macro-F1": val_f1})
        print(f"Epoch {epoch}: loss {train_loss:.3f} | validation accuracy {val_accuracy:.3f} | "
              f"macro-F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, best_weights, waited = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited >= patience:
                print(f"Stopped early: no improvement for {patience} epochs.")
                break

    model.load_state_dict(best_weights)
    return history


def load_model(path, device, num_classes=4):
    """Rebuilds the ResNet18 and loads saved weights into it."""
    model = build_model(num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device)


def fine_tune(model, train_loader, loss_fn, optimizer, device, epochs=5):
    """Trains for a fixed number of epochs, with no early stopping.

    Used in cross-validation, where the held-out fold is only for scoring and must not be
    used to choose when to stop. Returns the training loss per epoch.
    """
    return [train_one_epoch(model, train_loader, loss_fn, optimizer, device) for _ in range(epochs)]
