import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator

from Custom_layers import *
from utils import *

train_dataset = dsets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

test_dataset = dsets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=True
)


@variational_estimator
class Bayesian_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BayesianConv2d(1, 6, (5, 5), [1, 3, 3, 3])
        self.conv2 = BayesianConv2d(6, 16, (5, 5), [3, 8, 3, 3])
        self.fc1 = BayesianLinear(256, 120, 30, 20)
        self.fc2 = BayesianLinear(120, 84, 20, 5)
        self.fc3 = BayesianLinear(84, 10, 5, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Bayesian_Network().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


l = [
    module
    for module in classifier.modules()
    if isinstance(module, (BayesianLinear, BayesianConv2d, BayesianGRU))
]

pnet_size = 0
gnet_size = 0
for i in l:
    pnet_size += i.pnet_size
    gnet_size += i.gnet_size

print(f"Compression: {pnet_size/gnet_size:.2f}")

# Specify the name of the directory to create
dir_name = f"Results/CNN_compression_{pnet_size/gnet_size:.2f}"

# Use the makedirs() function to create the directory if it doesn't exist
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print("Directory", dir_name, "created")
else:
    print("Directory", dir_name, "already exists")


iteration = 0
test_acc = []
loss_arr = []
itr_arr = []
for epoch in range(100):
    for i, (datapoints, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        loss = classifier.sample_elbo(
            inputs=datapoints.to(device),
            labels=labels.to(device),
            criterion=criterion,
            sample_nbr=3,
            complexity_cost_weight=1 / 50000,
        )
        loss_arr.append(loss.item())
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 250 == 0:
            print(loss)
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data

                    outputs = classifier(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
            print(
                "Iteration: {} | Accuracy of the network on the 10000 test images: {} %".format(
                    str(iteration), str(100 * correct / total)
                )
            )

            test_acc.append(100 * correct / total)
            itr_arr.append(iteration)
            np.save(f"{dir_name}/test_acc", [test_acc, itr_arr])
            np.save(f"{dir_name}/loss_arr", loss_arr)
