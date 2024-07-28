import os
from datetime import datetime
import argparse
import random
import itertools
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import scipy.io as sio
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description="Main script to train an agent")
parser.add_argument("--array_id", type=int, default=0, help="ID for this array job")
args = parser.parse_args()


save_path = "./results"


date = datetime.now().strftime("%Y-%m-%d")
date_now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


# set hyperparameters for this experiment
seed_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ratio_arr = [30, 50, 150, 300, 600, 1000, 1500]
model_cf_arr = [
    "mean",
    "sampled_0",
    "sampled_1",
    "sampled_2",
    "sampled_3",
    "sampled_4",
    "sampled_5",
    "sampled_6",
    "sampled_7",
    "sampled_8",
    "sampled_9",
]

save_expected_arr = [True]


para_comb = list(itertools.product(seed_arr, ratio_arr, model_cf_arr))
(seed, ratio, model_cf) = para_comb[args.array_id - 1]


class MLP(torch.nn.Module):
    """Neural network class to predict MNIST digits based on image pixels"""

    def __init__(self, input_dim=784, hid_dim=800, out_dim=10):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim).to(self.device)
        self.fc2 = nn.Linear(hid_dim, out_dim).to(self.device)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def optimize_backprop(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def trainMNIST(
    model, model_name, device, train_loader, optimizer, epoch, epochs, test_loader=None
):

    correct_per_batch_test = []

    if test_loader != None:
        correct_per_batch_test.append(test(model, device, test_loader))

    print("Training " + model_name + "----------------------------------")

    correct = 0
    correct_per_batch = []
    for nepoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            correct_per_batch.append(
                pred.eq(target.view_as(pred)).sum().item() / pred.size()[0]
            )

            loss = F.cross_entropy(output, target)

            model.optimize_backprop(loss, optimizer)

            if test_loader != None:
                correct_per_batch_test.append(test(model, device, test_loader))

            if batch_idx % 10 == 0:
                print(
                    "Training "
                    + model_name
                    + " : Epoch={} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        print(
            "Training "
            + model_name
            + " : Epoch={} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )

    return correct, correct_per_batch, correct_per_batch_test


def test(model, device, test_loader):
    """Test MNISTN/CIFAR model"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return correct


if __name__ == "__main__":
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    number_of_epochs = 1

    home_path = "./results/2024-07-26"

    exp_folder = [
        name
        for name in os.listdir(home_path)
        if os.path.isdir(os.path.join(home_path, name))
    ]

    for exp in exp_folder:
        print(exp)
        if f"compression_{ratio}_" in exp and f"seed_{seed}" in exp:
            file_name = os.path.join(home_path, exp)
            with open(os.path.join(file_name, f"weights_{model_cf}.pkl"), "rb") as file:
                weights = pickle.load(file)

    mat_fname = os.path.join(file_name, f"{model_cf}_finetune_GPU_every.mat")

    CorrectPreTrainBatch = []
    CorrectPreTestBatch = []
    CorrectPreTrain = np.zeros(number_of_epochs)
    CorrectPreTest = np.zeros(number_of_epochs)

    device = torch.device("cuda")
    kwargs = {"num_workers": 1, "pin_memory": True}

    model = MLP().to(device)

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            weights_tensor = weights[name]["w"]
            biases_tensor = weights[name]["b"]

            # Set the loaded weights and biases in the MLP
            if model_cf == "mean":
                layer.weight.data = torch.transpose(torch.squeeze(weights_tensor), 0, 1)
            else:
                layer.weight.data = torch.squeeze(weights_tensor)

            layer.bias.data = torch.squeeze(biases_tensor)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    MNIST_train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    MNIST_test_dataset = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    test_set_length = len(MNIST_test_dataset)
    train_set_length = len(MNIST_train_dataset)
    remainder_length = train_set_length - test_set_length

    MNIST_validation_dataset, remainder = torch.utils.data.random_split(
        MNIST_train_dataset, [test_set_length, remainder_length]
    )

    train_loader_mnist = torch.utils.data.DataLoader(
        MNIST_train_dataset, batch_size=128, shuffle=True, **kwargs
    )
    test_loader_mnist = torch.utils.data.DataLoader(
        MNIST_test_dataset, batch_size=128, shuffle=True, **kwargs
    )
    valid_loader_mnist = torch.utils.data.DataLoader(
        MNIST_validation_dataset, batch_size=128, shuffle=True, **kwargs
    )

    for step_idx in range(number_of_epochs):
        CorrectPreTrain[step_idx], correct_per_batch_train, correct_per_batch_test = (
            trainMNIST(
                model,
                "MLP",
                device,
                train_loader_mnist,
                optimizer,
                step_idx,
                1,
                test_loader=test_loader_mnist,
            )
        )
        CorrectPreTrainBatch.append(correct_per_batch_train)
        CorrectPreTestBatch.append(correct_per_batch_test)

        d = {
            "CorrectPreTest": CorrectPreTest,
            "CorrectPreTrain": CorrectPreTrain,
            "CorrectPreTrainBatch": CorrectPreTrainBatch,
            "CorrectPreTestBatch": CorrectPreTestBatch,
            "test_set_length": test_set_length,
            "train_set_length": train_set_length,
        }

        sio.savemat(mat_fname, d)  # saves the dictionary only
