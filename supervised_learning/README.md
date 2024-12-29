Here is a README for running the provided script:

---

# Bayesian Neural Network Training Script

This script trains a Bayesian neural network (BNN) using the MNIST dataset, leveraging Bayesian linear layers with variational inference. It allows for hyperparameter tuning, saving model weights, and evaluating the model's performance.

## Requirements

### Dependencies

Install the required Python libraries:

```bash
pip install torch torchvision blitz numpy
```

Ensure you also have the following custom modules and layers in your project directory:
- `custom_layers.py`
- `utils.py`

### Hardware
- **GPU** (optional): Training is faster if a CUDA-compatible GPU is available.

## How to Run

### Step 1: Prepare the Environment
Ensure that the following directories exist or can be created:
- `data/`: The MNIST dataset will be downloaded here.
- `results/`: Model results and checkpoints will be saved here.

### Step 2: Set Up the Script
Adjust the script's default paths or parameters if needed:
- **Data path**: Use the `--data_path` argument to specify where the dataset is located or downloaded.
- **Results path**: Use the `--save_path` argument to specify where results will be saved.
- **Epochs**: Adjust the number of training epochs with the `--epochs` argument.

### Step 3: Execute the Script
Run the script using:

```bash
python script_name.py --array_id <job_id> --data_path <data_directory> --save_path <results_directory> --epochs <num_epochs>
```

Replace the placeholders with your desired values:
- `<job_id>`: Index for hyperparameter combinations.
- `<data_directory>`: Path to the dataset directory (default: `data/`).
- `<results_directory>`: Path to save results (default: `results/`).
- `<num_epochs>`: Number of epochs to train the model (default: `10`).

### Example

```bash
python script_name.py --array_id 0 --data_path ./data/ --save_path ./results/ --epochs 20
```

### Step 4: View Results
The script creates a timestamped directory under `results/` containing:
- `best_model.pth`: Best model weights based on validation accuracy.
- `model.pth`: Final model weights.
- `weights.pkl`: Sampled network weights.
- `weights_mean.pkl`: Mean network weights.
- `weights_sampled_<n>.pkl`: 10 sampled network weights.
- `val_acc.npy`: Validation accuracy for all epochs.
- `loss_arr.npy`: Loss values during training.
- `test_acc_compression_<ratio>_seed_<seed>.npy`: Test accuracy for the best model.

### Hyperparameter Configuration

The script uses combinations of:
- `seed_arr`: [0, 1, 2, 3, 4]
- `ratio_arr`: [2000]

The combination indexed by `--array_id` determines which hyperparameter set is used. Modify `seed_arr` or `ratio_arr` in the script to try different configurations.
