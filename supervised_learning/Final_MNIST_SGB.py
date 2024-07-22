import os
import argparse
import random
import itertools
import pickle
import numpy as np
from datetime import datetime
from utils import *
from Custom_layers import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import (
    TrainableRandomDistribution,
    PriorWeightDistribution,
)

#######################################################################################
###  Use job array number to index which combination of hyperparameters to use
#######################################################################################

# TODO: take as input manually defined hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int, default=0)
parser.add_argument("--data_path", type=str)
parser.add_argument("--save_path", type=str)

args = parser.parse_args()

seed_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ratio_arr = [30, 50, 150, 300, 600, 1000, 1500]

para_comb = list(itertools.product(seed_arr, ratio_arr))
(seed, ratio) = para_comb[args.array_id]


#######################################################################################
###  Calculate the number of hidden units based on desired compression ratio
#######################################################################################

pnet_size = 784 * 800 + 800 * 10
if ratio < 600:
    n_in_type = 30
elif ratio < 1500:
    n_in_type = 20
else:
    n_in_type = 10

n_hid_type = int(np.round((pnet_size / ((n_in_type + 10) * ratio))))
print(f"Compression ratio: {ratio}, n_hid_type: {n_hid_type}")


#######################################################################################
###  Set constants (seed, path to load data, and path to save results)
#######################################################################################

data_path = args.data_path
save_path = args.save_path
date = datetime.now().strftime("%Y-%m-%d")
date_now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

# Specify the name of the directory to create
dir_name = os.path.join(
    save_path,
    f"{date_now}_compression_{ratio}_in_{n_in_type}_hid_{n_hid_type}_seed_{seed}",
)

# Use the makedirs() function to create the directory if it doesn't exist
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print("Directory", dir_name, "created")
else:
    print("Directory", dir_name, "already exists")

# Set seed for this experiment
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#######################################################################################
###  Prepare dataset
#######################################################################################

train_dataset = dsets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)


test_dataset = dsets.MNIST(
    root=data_path,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)


test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=True
)


#######################################################################################
###  Construct the WS-BNN model class
#######################################################################################


@variational_estimator
class Bayesian_network(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = BayesianLinear(784, 800, n_in_type, n_hid_type, WS=True)
        self.fc2 = BayesianLinear(800, 10, n_hid_type, 10, WS=True)

    def forward(self, x):
        out = x
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


#######################################################################################
###  Define model and optimizers
#######################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Bayesian_network().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

l = [module for module in classifier.modules() if isinstance(module, BayesianLinear)]

#######################################################################################
###  Model training
#######################################################################################

test_acc = []
loss_arr = []
iteration = 0
for epoch in range(100):
    for i, (datapoints, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        datapoints = datapoints.flatten(1, -1)
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

        # Evaluate test accuracy on MNIST and save accuracy & loss after every 200 gradient steps
        iteration += 1
        if iteration % 200 == 0:
            print(loss)
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.flatten(1, -1)
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

            np.save(f"{dir_name}/test_acc", test_acc)
            np.save(f"{dir_name}/loss_arr", loss_arr)


#######################################################################################
###  Save trained model and sampled network weights
#######################################################################################

# Save the model
torch.save(classifier.state_dict(), os.path.join(dir_name, "model.pth"))

# Save an instance of the sampled weights of the network
weights = {}
name = ["fc1", "fc2"]
for i, layer in enumerate(l):
    weights[name[i]] = {}
    weights[name[i]]["w"], weights[name[i]]["b"] = layer.sample_layer_weights(
        expected_value=False
    )

with open(os.path.join(dir_name, "weights.pkl"), "wb") as f:
    pickle.dump(weights, f)


# saves the mean network

for i, layer in enumerate(l):
    weights[name[i]] = {}
    weights[name[i]]["w"], weights[name[i]]["b"] = layer.sample_layer_weights(
        expected_value=True
    )


with open(os.path.join(dir_name, "weights_mean.pkl"), "wb") as f:
    pickle.dump(weights, f)

# save the model
torch.save(classifier.state_dict(), os.path.join(dir_name, "model.pth"))

# saves 10 sampled 10 networks
for model_num_samp in range(10):
    for i, layer in enumerate(l):
        weights[name[i]] = {}
        weights[name[i]]["w"], weights[name[i]]["b"] = layer.sample_layer_weights(
            expected_value=False
        )

    with open(
        os.path.join(dir_name, f"weights_sampled_{model_num_samp}.pkl"), "wb"
    ) as f:
        pickle.dump(weights, f)
