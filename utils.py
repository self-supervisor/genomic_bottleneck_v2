import torch
import numpy as np


def find_num_neuron_per_type(num_features, num_types):
    features = int(num_features)
    types = int(num_types)

    # Calculate the average value of each element
    avg_value = int(features // types)
    # Initialize the array with elements of value avg_value
    arr = np.ones(types) * avg_value

    # Distribute the remaining features across the elements
    remainder = features - sum(arr)
    arr[: int(remainder)] += 1

    return arr.astype(np.int32)


def get_indices(pre_unit_types, n_pre_neurons, post_unit_types, n_post_neurons):
    pre_type_indices = torch.tile(
        torch.tensor(pre_unit_types[None, :, None]), (1, 1, n_post_neurons)
    )

    post_type_indices = torch.tile(
        torch.tensor(post_unit_types[None, :, None]), (1, 1, n_pre_neurons)
    )

    indices = torch.stack(
        (pre_type_indices, post_type_indices.permute(0, 2, 1)), axis=-1
    )

    return indices


def to1d(x, y, z, num_inp, num_out):
    return (z * num_inp * num_out) + (y * num_out) + x


def to1d_CNN(a, x, y, z, out_channel, in_channel, num_out, num_inp):
    return (
        (a * num_inp * in_channel * num_out)
        + (x * num_inp * num_out)
        + (y * num_inp)
        + z
    )


def gather2D(tensor, indices):

    old_shape = indices.shape[:-1]
    tensor = gather_nd_torch(tensor, indices, batch_dim=1)
    return torch.reshape(tensor, old_shape)


def gather_nd_torch(params, indices, batch_dim=0):

    expand = batch_dim == 0
    if expand:
        params = torch.unsqueeze(params, 0)
        indices = torch.unsqueeze(indices, 0)
        batch_dim = 1

    batch_dims = params.size()[:batch_dim]  # [b1, ..., bn]
    batch_size = np.cumprod(list(batch_dims))[-1]  # b1 * ... * bn
    c_dim = params.size()[-1]  # c
    grid_dims = params.size()[batch_dim:-1]  # [g1, ..., gm]
    n_indices = indices.size()[batch_dim:-1]  # x
    org_indices = indices
    n_indices = np.cumprod(list(n_indices))[-1]
    n_pos = indices.size(-1)  # m
    try:
        last_dim = params.size()[batch_dim + n_pos :][0]
    except:
        last_dim = 1
    # reshape leadning batch dims to a single batch dim
    params = params.reshape(batch_size, *grid_dims, c_dim)
    indices = indices.reshape(batch_size, n_indices, n_pos)
    idx_arr = [[] for _ in range(n_pos + 1)]
    for i in range(batch_size):
        idx_arr[0].append([i for _ in range(len(indices[i][:, 0]))])
        for j in range(n_pos):
            idx_arr[j + 1].append(indices[i][:, j].numpy())

    gather_dims = idx_arr

    try:
        gather_dims_flat = to1d_CNN(
            np.array(gather_dims[1]),
            np.array(gather_dims[2]),
            np.array(gather_dims[3]),
            np.array(gather_dims[4]),
            (gather_dims[1][0][-1] + 1),
            (gather_dims[2][0][-1] + 1),
            (gather_dims[3][0][-1] + 1),
            (gather_dims[4][0][-1] + 1),
        )
    except:
        gather_dims_flat = to1d(
            np.array(gather_dims[2]),
            np.array(gather_dims[1]),
            np.array(gather_dims[0]),
            (gather_dims[1][-1] + 1),
            (gather_dims[2][-1] + 1),
        )

    gathered = torch.index_select(
        params.flatten(),
        0,
        torch.tensor(gather_dims_flat.flatten()).to(params.flatten().device),
    )

    gathered = gathered.reshape(
        *batch_dims, *org_indices.size()[batch_dim:-1], last_dim
    )

    gathered = torch.squeeze(gathered)

    return gathered


def get_indices_CNN(n_neurons_arr, unit_types_arr):
    n_pre_neurons = n_neurons_arr[3]
    n_post_neurons = n_neurons_arr[2]

    n_pre_channels = n_neurons_arr[1]
    n_post_channels = n_neurons_arr[0]

    pre_unit_types = unit_types_arr[3]
    post_unit_types = unit_types_arr[2]
    pre_channel_types = unit_types_arr[1]
    post_channel_types = unit_types_arr[0]

    pre_type_indices = torch.tile(
        torch.tensor(pre_unit_types[None, None, None, :]),
        (1, n_post_channels, n_pre_channels, n_post_neurons, 1),
    )

    post_type_indices = torch.tile(
        torch.tensor(post_unit_types[None, None, :, None]),
        (1, n_post_channels, n_pre_channels, 1, n_pre_neurons),
    )

    pre_channel_indices = torch.tile(
        torch.tensor(pre_channel_types[None, :, None, None]),
        (1, n_post_channels, 1, n_post_neurons, n_pre_neurons),
    )

    post_channel_indices = torch.tile(
        torch.tensor(post_channel_types[:, None, None, None]),
        (1, 1, n_pre_channels, n_post_neurons, n_pre_neurons),
    )

    indices = torch.stack(
        (
            post_channel_indices,
            pre_channel_indices,
            post_type_indices,
            pre_type_indices,
        ),
        axis=-1,
    )

    return indices
