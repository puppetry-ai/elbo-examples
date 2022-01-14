import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

import elbo.elbo
from elbo.elbo import ElboModel


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class MNISTClassifier(ElboModel, nn.Module):
    def get_artifacts_directory(self):
        return 'artifacts'

    def save_state(self):
        model_path = os.path.join(self.get_artifacts_directory(), "mnist_model")
        torch.save(self.state_dict(), model_path)
        print(f"Saving model to {model_path}")

    def load_state(self, state_dir):
        model_path = os.path.join(self.get_artifacts_directory(), "mnist_model")
        print(f"Loading model from {model_path}")
        self.load_state_dict(torch.load(model_path))

    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 10),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.net(x)
        return out


def train(model, train_dataset, batch_size=2000, lr=0.001):
    device = get_device()
    print(f"Training running on device - {device}")
    model.to(device)

    model.train()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    loss = None
    for index, (x, y) in tqdm.tqdm(enumerate(train_loader)):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)

        loss = F.nll_loss(y_hat, y)
        loss.backward()
        optimizer.step()

    return loss.detach().cpu().numpy()


def test(model, test_dataset):
    device = get_device()
    model.to(device)
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test_dataset)
        loss = 0.0
        for index, (x, y) in tqdm.tqdm(enumerate(test_loader)):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss += F.nll_loss(y_hat, y, reduction='sum').detach().cpu().numpy()

        loss = loss / len(test_dataset)
        print(f"Average Test Loss = {loss}")


if __name__ == '__main__':
    print(f"Training MNIST classifier")
    _train_data = datasets.MNIST("data", train=True, transform=transforms.ToTensor(), download=True)
    _test_data = datasets.MNIST("data", train=False, transform=transforms.ToTensor(), download=True)
    _model = MNISTClassifier()
    _num_epochs = 10

    for _epoch in elbo.elbo.ElboEpochIterator(range(0, _num_epochs), _model, save_state_interval=1):
        _loss = train(_model, _train_data)
        print(f"Epoch = {_epoch} Loss = {_loss}")

    test(_model, _test_data)
