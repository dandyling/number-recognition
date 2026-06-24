"""Train a small CNN to recognise handwritten digits (0-9) on the MNIST dataset.

The dataset is downloaded automatically from the internet via torchvision.
The trained weights are saved to ``model/mnist_cnn.pt`` so the backend can load
them at inference time.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
WEIGHTS_PATH = os.path.join(HERE, "mnist_cnn.pt")

# Mean/std of the MNIST training set. Shared with the backend so that images
# uploaded through the UI are normalised exactly the way the model was trained.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class Net(nn.Module):
    """A compact convolutional network that reaches ~99% test accuracy."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_loaders(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def main(epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders()
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                    f"loss={loss.item():.4f}"
                )
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} test accuracy: {acc * 100:.2f}%")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
