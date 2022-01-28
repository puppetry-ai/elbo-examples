import torch
import torch.nn as nn
import torch.optim as optim
from ray.train import torch as raytorch
from ray.train import Trainer
import elbo.elboray
num_samples = 20
input_size = 10
layer_size = 15
output_size = 5


class NeuralNetwork(nn.Module):
    """
    Neural network example based on -- https://docs.ray.io/en/latest/train/train.html
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, layer_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(layer_size, output_size)

    def forward(self, input):
        return self.layer2(self.relu(self.layer1(input)))


def train_func_distributed(model_input, model_labels):
    num_epochs = 3
    model = NeuralNetwork()
    model = raytorch.prepare_model(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        output = model(model_input)
        loss = loss_fn(output, model_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")


def run():
    trainer = Trainer(backend="torch", num_workers=4)
    trainer.start()
    results = trainer.run(train_func_distributed)
    trainer.shutdown()
    print(results)


if __name__ == "__main__":
    run()
