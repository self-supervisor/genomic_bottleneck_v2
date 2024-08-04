"""
    003_MNIST_automatic_2.py

    This is a demo file for geneNets2_0
    This file shows how to encode MNIST via the bottleneck by automatically setting up
    G-net networks. In this case the last two layers of the network are changed using 
    updateLayer function to yield one-hot vector as an output. The biases for the last
    layer are not learned but directly encoded in the weights of a one-layer g-net
    
    questions -- ask Alex (akula@cshl.edu)
    
"""

import argparse
import random
import math as math
import os
import itertools
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

import geneNets2_0 as gn

parser = argparse.ArgumentParser(description="Main script to train an agent")
parser.add_argument("--array_id", type=int, default=0, help="ID for this array job")
args = parser.parse_args()

seed_arr = [4, 5, 6, 7, 8, 9]
ratio_arr = [30, 50, 150, 300, 600, 1000, 1500]

para_comb = list(itertools.product(seed_arr, ratio_arr))
(seed, ratio) = para_comb[args.array_id]


def numberOfParameters(model):
    numPar = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            numPar = numPar + np.array(parameter.data.shape).prod()
    return numPar


class MNISTN(nn.Module):
    """Neural network class to predict MNIST digits based on image pixels"""

    def __init__(self):
        super(MNISTN, self).__init__()
        self.fc = nn.Linear(28 * 28, 800, bias=True)
        self.fc2 = nn.Linear(800, 10, bias=True)
        self.numberOfParameters = numberOfParameters(self)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, loss_type, epoch, max_batch_idx=100):
    """Train either of the models"""
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if loss_type == "nll":
            loss = F.cross_entropy(output, target)
        if loss_type == "mse":
            loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        if loss_type == "nll" and batch_idx == max_batch_idx:
            break
        if loss_type == "mse" and batch_idx == max_batch_idx:
            break


def test(model, device, test_loader):
    """Test MNISTN model"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
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


def main():

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    correct_max = 0
    max_epochs = 500
    number_of_steps = 100

    kwargs = {"num_workers": 4, "pin_memory": True}

    start_date = datetime.now().strftime("%Y-%m-%d")
    home_path = "./results"
    file_name = os.path.join(
        home_path,
        start_date,
        "MNIST_seed_" + str(seed) + "_ratio_" + str(ratio) + "_003",
    )

    save_path = os.path.join(home_path, start_date)
    os.makedirs(save_path, exist_ok=True)

    fname2load = os.path.join(home_path, file_name)
    fname2save = os.path.join(save_path, file_name + "_MLP_pretrained")
    mat_fname = fname2save + ".mat"
    mod_fname = fname2save + ".pt"
    load_mod_fname = fname2load + ".pt"

    print(fname2save)

    CorrectTest = np.zeros(max_epochs)
    CorrectTrain = np.zeros(max_epochs)

    CorrectPostTest = np.zeros(number_of_steps)
    CorrectPostTrain = np.zeros(number_of_steps)

    # Define the neural networks
    device = torch.device("cuda")

    model_MNISTN = MNISTN().to(device)
    optimizer_MNISTN = optim.Adam(model_MNISTN.parameters())

    pnet_size = model_MNISTN.numberOfParameters

    n_hidden_units = int(np.round((pnet_size - 20 * ratio) / (60 * ratio)))
    print(f"Compression ratio: {ratio}, num hidden units: {n_hidden_units}")

    GNets = gn.GNetList(model_MNISTN, n_hidden_units)

    print("*" * 50)
    print(GNets)
    #
    #   The last layer needs to be corrected to use one-hot vectors as outputs
    #

    GNets.updateParameter(
        "fc2.weight",
        (gn.BIN, gn.HOT),
        [800, 10],
        [10, 10],
        numGDNLayers=2,
        hidden=(30, 10),
    )
    GNets.updateParameter(
        "fc2.bias", (gn.HOT), [10], [10], numGDNLayers=-1
    )  # for direct storage

    print(GNets.GNET_INPUTS[-1])

    GNets.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        useMultipleGPUs = True
        device_ids = [1, 2, 3, 4]
    else:
        useMultipleGPUs = False

    if useMultipleGPUs:
        GNets.parallelize()

    mnist_train_data = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    mnist_test_data = datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    test_set_length = len(mnist_test_data)
    train_set_length = len(mnist_train_data)
    remainder_length = train_set_length - test_set_length

    mnist_valid_data, remainder = torch.utils.data.random_split(
        mnist_train_data, [test_set_length, remainder_length]
    )

    train_loader_mnist = torch.utils.data.DataLoader(
        mnist_train_data, batch_size=100, shuffle=True, **kwargs
    )
    test_loader_mnist = torch.utils.data.DataLoader(
        mnist_test_data, batch_size=100, shuffle=True, **kwargs
    )
    valid_loader_mnist = torch.utils.data.DataLoader(
        mnist_valid_data, batch_size=100, shuffle=True, **kwargs
    )

    #
    # Training
    #

    for epoch in np.arange(max_epochs):
        print("***************************")
        train(model_MNISTN, device, train_loader_mnist, optimizer_MNISTN, "nll", epoch)

        GNets.getWeights(model_MNISTN)
        GNets.trainGNets(
            device, epoch, "default", (1, 3)
        )  # number of epochs to train 1 - weights, 3 - biases
        eps = 1.0
        GNets.generateWeights(model_MNISTN, device, epsilon=eps)

        print(GNets.numberOfParameters(model_MNISTN))
        print(GNets.gnetParameters())
        print(GNets.compression(model_MNISTN))

        #
        #   Testing new model
        #

        correct = test(model_MNISTN, device, test_loader_mnist)
        CorrectTest[epoch] = correct

        correct = test(model_MNISTN, device, valid_loader_mnist)
        CorrectTrain[epoch] = correct

        # Save the best model
        if CorrectTrain[epoch] > correct_max:  # use validation results
            print("Saving model...\n")
            correct_max = correct

            GNets.saveAll(mod_fname, model_MNISTN, optimizer_MNISTN)

        sio.savemat(
            mat_fname,
            {
                "W1": GNets.extractWeights("fc.weight").cpu().data.numpy(),
                "W2": GNets.extractWeights("fc2.weight").cpu().data.numpy(),
                "b1": GNets.extractWeights("fc.bias").cpu().data.numpy(),
                "CorrectTest": CorrectTest,
                "CorrectTrain": CorrectTrain,
            },
        )

    #
    #   Testing pretrained networks
    #

    print("Testing pretrained networks")

    for step_idx in range(number_of_steps):
        CorrectPostTest[step_idx] = test(model_MNISTN, device, test_loader_mnist)
        CorrectPostTrain[step_idx] = test(model_MNISTN, device, valid_loader_mnist)

        train(
            model_MNISTN,
            device,
            train_loader_mnist,
            optimizer_MNISTN,
            "nll",
            step_idx,
            max_batch_idx=10,
        )

    sio.savemat(
        mat_fname,
        {
            "W1": GNets.extractWeights("fc.weight").cpu().data.numpy(),
            "W2": GNets.extractWeights("fc2.weight").cpu().data.numpy(),
            "b1": GNets.extractWeights("fc.bias").cpu().data.numpy(),
            "CorrectTest": CorrectTest,
            "CorrectTrain": CorrectTrain,
            "CorrectPostTest": CorrectPostTest,
            "CorrectPostTrain": CorrectPostTrain,
        },
    )

    return model_MNISTN


if __name__ == "__main__":
    model_MNISTN = main()
