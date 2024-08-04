import argparse
import itertools
import os
import pickle
import random
from datetime import datetime
import glob
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from blitz.losses import kl_divergence_from_nn
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import (
    PriorWeightDistribution,
    TrainableRandomDistribution,
)
from blitz.utils import variational_estimator
from custom_layers import *
from utils import *

#######################################################################################
###  Use job array number to index which combination of hyperparameters to use
#######################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int, default=0)
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--save_path", type=str, default="results/")
parser.add_argument("--epochs", type=int, default=10)

args = parser.parse_args()

seed_arr = list(range(5))
ratio_arr = [2000]
EPOCHS = args.epochs

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

#######################################################################################
###  Check for existing results and overwrite if incomplete
#######################################################################################

expected_dir_pattern = os.path.join(
    save_path, f"*_compression_{ratio}_in_{n_in_type}_hid_{n_hid_type}_seed_{seed}"
)

matching_dirs = glob.glob(expected_dir_pattern)

if matching_dirs:
    dir_to_check = matching_dirs[0]

    mean_network_exists = os.path.exists(os.path.join(dir_to_check, "weights_mean.pkl"))

    sampled_networks_exist = all(
        os.path.exists(os.path.join(dir_to_check, f"weights_sampled_{i}.pkl"))
        for i in range(10)
    )

    if mean_network_exists and sampled_networks_exist:
        print(
            f"Complete results already exist for compression {ratio}, seed {seed}. Skipping."
        )
        exit(0)
    else:
        print(
            f"Incomplete results found for compression {ratio}, seed {seed}. Removing and rerunning."
        )
        shutil.rmtree(dir_to_check)

dir_name = os.path.join(
    save_path,
    f"{date_now}_compression_{ratio}_in_{n_in_type}_hid_{n_hid_type}_seed_{seed}",
)

os.makedirs(dir_name, exist_ok=True)
print(f"Directory {dir_name} created or already exists")

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

# Split the dataset into train and validation
from torch.utils.data import random_split

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=64, shuffle=False
)

test_dataset = dsets.MNIST(
    root=data_path,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=False
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
###  Define evaluation function
#######################################################################################


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.flatten(1, -1)
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    accuracy = 100 * correct / total
    return accuracy


#######################################################################################
###  Model training
#######################################################################################

val_acc = []
loss_arr = []
best_val_accuracy = 0
best_model_state = None

for epoch in range(EPOCHS):
    classifier.train()
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

    # Evaluate on validation set
    val_accuracy = evaluate_model(classifier, val_loader, device)
    print(f"Epoch {epoch+1}/{EPOCHS} | Validation Accuracy: {val_accuracy:.2f}%")

    # Check if this is the best accuracy so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = classifier.state_dict()

        # Save the best model immediately
        torch.save(best_model_state, os.path.join(dir_name, "best_model.pth"))
        print(
            f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%"
        )

    val_acc.append(val_accuracy)
    np.save(f"{dir_name}/val_acc", val_acc)
    np.save(f"{dir_name}/loss_arr", loss_arr)

#######################################################################################
###  Save trained model and sampled network weights
#######################################################################################

# Load and save the best model
if best_model_state is not None:
    classifier.load_state_dict(best_model_state)
    torch.save(classifier.state_dict(), os.path.join(dir_name, "model.pth"))
    print(f"Final model saved with best validation accuracy: {best_val_accuracy:.2f}%")

    # Evaluate the best model on the test set
    test_accuracy = evaluate_model(classifier, test_loader, device)
    print(f"Best model performance on test set: {test_accuracy:.2f}%")
    np.save(f"{dir_name}/test_acc_compression_{ratio}_seed_{seed}.npy", test_accuracy)
else:
    torch.save(classifier.state_dict(), os.path.join(dir_name, "model.pth"))
    print("Final model saved (no improvement found during training)")

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

# saves 10 sampled networks
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
