import torch
from Custom_layers import BayesianLinear


def test_number_of_cell_types_floors():
    layer = BayesianLinear(
        in_features=2, out_features=4, neuron_types_in=64, neuron_types_out=64
    )
    assert layer.weight_sampler.mu.shape == torch.Size([1, 2, 4])
    assert layer.weight_sampler.rho.shape == torch.Size([1, 2, 4])
