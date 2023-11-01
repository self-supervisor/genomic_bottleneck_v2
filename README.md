# 🧬 Genomic Bottleneck v2

This project aims at training a Bayesian neural network version of the genomic bottleneck for Reinforcement Learning (RL) tasks, with a primary focus on continuous control utilizing the high-throughput Brax simulator. The training code has been adapted from [this notebook 🔗](https://github.com/google/brax/blob/main/notebooks/training_torch.ipynb).

## Getting Started

### Prerequisites

Ensure you have `conda` installed on your machine. You can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

### Installation

This is what worked on Elzar. It might be different for you depending on your CUDA versions.

1. Create and activate a new conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate brax-torch-reprod
    ```

2. Install the necessary additional packages:
    ```bash
    pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.7+cuda11.cudnn82-cp311-cp311-manylinux2014_x86_64.whl
    pip install blitz-bayesian-pytorch
    ```

## Usage

### Quick Start

Run a single example (approximately 20 minutes):
```bash
python training_torch.py --config-name=ant eval_population=False
```

To get phenotypic diversity plot:

```bash
python training_torch.py --config-name ant
```

### Reproducing Plots

To sweep through different compressions and environments run:

```bash
qsub UGE_job.sh
```

Then edit the code in plot.py to go through the CSVs and plot them.

Code for Bayesian bottleneck layes was given to me by Divyansha.

README generated with GPT-4.
