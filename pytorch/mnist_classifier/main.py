import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

import elbo.elbo
from elbo.elbo import ElboModel

import wandb

wandb.login(key='554f1579018d2fa355625a4c811986e7bc959059')
wandb.init(project="elbo-mnist-classifier", entity="elbo")


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


def train(model, train_dataset, batch_size=200, lr=0.001):
    device = get_device()
    model.to(device)

    model.train()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    loss = None
    y_output = []
    for index, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)

        loss = F.nll_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        y_output.append(y_hat)

    return loss.detach().cpu().numpy(), y_output


def test(model, test_dataset):
    device = get_device()
    model.to(device)
    y_output = []
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test_dataset)
        loss = 0.0
        for index, (x, y) in tqdm.tqdm(enumerate(test_loader)):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            y_output.append(y_hat)
            loss += F.nll_loss(y_hat, y, reduction='sum').detach().cpu().numpy()

        loss = loss / len(test_dataset)
        wandb.log({"test_loss": loss})
        print(f"Average Test Loss = {loss}")
        return y_output


if __name__ == '__main__':
    print(f"Training MNIST classifier on {get_device()}")
    _train_data = datasets.MNIST("data", train=True, transform=transforms.ToTensor(), download=True)
    _test_data = datasets.MNIST("data", train=False, transform=transforms.ToTensor(), download=True)

    _model = MNISTClassifier()
    _num_epochs = 100
    _batch_size = 2000
    _lr = 0.01

    wandb.config = {
        "learning_rate": _lr,
        "epochs": _num_epochs,
        "batch_size": _batch_size
    }

    wandb.watch(_model)
    _device = get_device()

    y_test_true = _test_data.test_labels.to(_device)
    y_train_true = _train_data.train_labels.to(_device)

    for _epoch in elbo.elbo.ElboEpochIterator(range(0, _num_epochs), _model, save_state_interval=10):
        _loss, y_train_pred = train(_model, _train_data, _batch_size, _lr)

        model_output = test(_model, _test_data)
        with torch.no_grad():
            y_test_pred = torch.argmax(torch.cat(model_output), dim=1)
            y_train_pred = torch.argmax(torch.cat(y_train_pred), dim=1)

            test_accuracy = 100. * (torch.sum(y_test_pred == y_test_true).detach().cpu() / (_test_data.data.size(0)))
            train_accuracy = 100. * (torch.sum(y_train_pred == y_train_true).detach().cpu() / (_train_data.data.size(0)))
            wandb.log({"loss": _loss, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy})
            print(f"Train accuracy - {train_accuracy} % Test accuracy - {test_accuracy} % Loss = {_loss}")