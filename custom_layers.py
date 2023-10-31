import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from blitz.losses import kl_divergence_from_nn
from blitz.modules.base_bayesian_module import BayesianModule, BayesianRNN
from blitz.modules.weight_sampler import (
    PriorWeightDistribution,
    TrainableRandomDistribution,
)
from blitz.utils import variational_estimator

from utils import *


class TrainableRandomDistribution_weight_share(nn.Module):
    def __init__(self, mu, rho, indices):
        super().__init__()

        self.weight_mu_share = mu
        self.weight_rho_share = rho
        self.indices = indices

        self.indices.flat = gather_nd_torch_dims_flat(
            self.weight_mu_share, self.indices, batch_dim=1
        )

        self.mu = gather2D(self.weight_mu_share, self.indices, self.indices.flat)
        self.rho = gather2D(self.weight_rho_share, self.indices, self.indices.flat)
        self.register_buffer("eps_w", torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is
        a function from a trainable parameter, and adding a mean
        sets those weights as the current ones
        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.mu = gather2D(self.weight_mu_share, self.indices, self.indices.flat)
        self.rho = gather2D(self.weight_rho_share, self.indices, self.indices.flat)

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w[0].permute(1, 0)

    def log_posterior(self, w=None):
        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost
        returns:
            torch.tensor with shape []
        """

        assert (
            self.w is not None
        ), "You can only have a log posterior for W if you've already sampled it"
        if w is None:
            w = self.w

        log_sqrt2pi = np.log(np.sqrt(2 * self.pi))
        log_posteriors = (
            -log_sqrt2pi
            - torch.log(self.sigma)
            - (((w - self.mu) ** 2) / (2 * self.sigma ** 2))
            - 0.5
        )
        return log_posteriors.sum()


class TrainableRandomDistribution_weight_share_CNN(nn.Module):
    def __init__(self, mu, rho, indices):
        super().__init__()

        self.weight_mu_share = mu
        self.weight_rho_share = rho
        self.indices = indices

        self.indices.flat = gather_nd_torch_dims_flat(
            self.weight_mu_share, self.indices, batch_dim=1
        )

        self.mu = gather2D(self.weight_mu_share, self.indices, self.indices.flat)
        self.rho = gather2D(self.weight_rho_share, self.indices, self.indices.flat)
        self.register_buffer("eps_w", torch.Tensor(self.mu.shape))
        self.sigma = None
        self.w = None
        self.pi = np.pi

    def sample(self):
        """
        Samples weights by sampling form a Normal distribution, multiplying by a sigma, which is
        a function from a trainable parameter, and adding a mean
        sets those weights as the current ones
        returns:
            torch.tensor with same shape as self.mu and self.rho
        """

        self.mu = gather2D(self.weight_mu_share, self.indices, self.indices.flat)
        self.rho = gather2D(self.weight_rho_share, self.indices, self.indices.flat)

        self.eps_w.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.mu + self.sigma * self.eps_w
        return self.w[0]

    def log_posterior(self, w=None):
        """
        Calculates the log_likelihood for each of the weights sampled as a part of the complexity cost
        returns:
            torch.tensor with shape []
        """

        assert (
            self.w is not None
        ), "You can only have a log posterior for W if you've already sampled it"
        if w is None:
            w = self.w

        log_sqrt2pi = np.log(np.sqrt(2 * self.pi))
        log_posteriors = (
            -log_sqrt2pi
            - torch.log(self.sigma)
            - (((w - self.mu) ** 2) / (2 * self.sigma ** 2))
            - 0.5
        )
        return log_posteriors.sum()


class BayesianGRU(BayesianRNN):
    """
    Bayesian GRU layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).
    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        neuron_types_in: int -> number of input type neurons
        neuron_types_out: int -> number of output type neurons
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """

    def __init__(
        self,
        in_features,
        out_features,
        neuron_types_in,
        neuron_types_out,
        WS=True,
        bias=True,
        prior_sigma_1=0.1,
        prior_sigma_2=0.002,
        prior_pi=1,
        posterior_mu_init=0,
        posterior_rho_init=-6.0,
        freeze=False,
        prior_dist=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.freeze = freeze

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        self.neuron_types_in = neuron_types_in
        self.neuron_types_out = neuron_types_out
        self.WS = WS
        if self.WS == False:
            # Variational weight parameters and sample for weight ih
            self.weight_ih_mu = nn.Parameter(
                torch.Tensor(in_features, out_features * 4).normal_(
                    posterior_mu_init, 0.1
                )
            )
            self.weight_ih_rho = nn.Parameter(
                torch.Tensor(in_features, out_features * 4).normal_(
                    posterior_rho_init, 0.1
                )
            )
            self.weight_ih_sampler = TrainableRandomDistribution(
                self.weight_ih_mu, self.weight_ih_rho
            )
            self.weight_ih = None

            # Variational weight parameters and sample for weight hh
            self.weight_hh_mu = nn.Parameter(
                torch.Tensor(out_features, out_features * 4).normal_(
                    posterior_mu_init, 0.1
                )
            )
            self.weight_hh_rho = nn.Parameter(
                torch.Tensor(out_features, out_features * 4).normal_(
                    posterior_rho_init, 0.1
                )
            )
            self.weight_hh_sampler = TrainableRandomDistribution(
                self.weight_hh_mu, self.weight_hh_rho
            )
            self.weight_hh = None

        else:
            self.pnet_size = (
                in_features * out_features * 4 + out_features * out_features * 4
            )
            self.gnet_size = (
                neuron_types_in * neuron_types_out * 4
                + neuron_types_out * neuron_types_out * 4
            )

            n_in_neurons_per_type = find_num_neuron_per_type(
                in_features, self.neuron_types_in
            )
            in_neuron_type_ids = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_in_neurons_per_type)], axis=0
            ).astype(np.int32)

            n_out_neurons_per_type = find_num_neuron_per_type(
                out_features * 4, self.neuron_types_out * 4
            )
            out_neuron_type_ids = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_out_neurons_per_type)], axis=0
            ).astype(np.int32)

            indices = get_indices(
                in_neuron_type_ids, in_features, out_neuron_type_ids, out_features * 4
            )
            # Variational weight parameters and sample for weight ih
            self.weight_ih_mu = nn.Parameter(
                torch.Tensor(
                    1, self.neuron_types_out * 4, self.neuron_types_in
                ).normal_(posterior_mu_init, 0.1)
            )
            self.weight_ih_rho = nn.Parameter(
                torch.Tensor(
                    1, self.neuron_types_out * 4, self.neuron_types_in
                ).normal_(posterior_rho_init, 0.1)
            )
            self.weight_ih_sampler = TrainableRandomDistribution_weight_share(
                self.weight_ih_mu, self.weight_ih_rho, indices
            )
            self.weight_ih = None

            n_in_neurons_per_type = find_num_neuron_per_type(
                out_features, self.neuron_types_out
            )
            in_neuron_type_ids = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_in_neurons_per_type)], axis=0
            ).astype(np.int32)

            n_out_neurons_per_type = find_num_neuron_per_type(
                out_features * 4, self.neuron_types_out * 4
            )
            out_neuron_type_ids = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_out_neurons_per_type)], axis=0
            ).astype(np.int32)

            indices = get_indices(
                in_neuron_type_ids, out_features, out_neuron_type_ids, out_features * 4
            )

            # Variational weight parameters and sample for weight hh
            self.weight_hh_mu = nn.Parameter(
                torch.Tensor(
                    1, self.neuron_types_out * 4, self.neuron_types_out
                ).normal_(posterior_mu_init, 0.1)
            )
            self.weight_hh_rho = nn.Parameter(
                torch.Tensor(
                    1, self.neuron_types_out * 4, self.neuron_types_out
                ).normal_(posterior_rho_init, 0.1)
            )
            self.weight_hh_sampler = TrainableRandomDistribution_weight_share(
                self.weight_hh_mu, self.weight_hh_rho, indices
            )
            self.weight_hh = None

        # Variational weight parameters and sample for bias
        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features * 4).normal_(posterior_mu_init, 0.1)
        )
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features * 4).normal_(posterior_rho_init, 0.1)
        )
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
        self.bias = None

        # our prior distributions
        self.weight_ih_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.weight_hh_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.bias_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )

        self.init_sharpen_parameters()

        self.log_prior = 0
        self.log_variational_posterior = 0

    def sample_weights(self):
        # sample weights
        weight_ih = self.weight_ih_sampler.sample()
        weight_hh = self.weight_hh_sampler.sample()
        if self.WS:
            weight_ih = weight_ih.permute(1, 0)
            weight_hh = weight_hh.permute(1, 0)

        # if use bias, we sample it, otherwise, we are using zeros
        if self.use_bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = 0
            b_log_posterior = 0
            b_log_prior = 0

        bias = b

        # gather weights variational posterior and prior likelihoods
        self.log_variational_posterior = (
            self.weight_hh_sampler.log_posterior()
            + b_log_posterior
            + self.weight_ih_sampler.log_posterior()
        )

        self.log_prior = (
            self.weight_ih_prior_dist.log_prior(weight_ih)
            + b_log_prior
            + self.weight_hh_prior_dist.log_prior(weight_hh)
        )

        self.ff_parameters = [weight_ih, weight_hh, bias]
        return weight_ih, weight_hh, bias

    def get_frozen_weights(self):
        # get all deterministic weights
        weight_ih = self.weight_ih_mu
        weight_hh = self.weight_hh_mu
        if self.use_bias:
            bias = self.bias_mu
        else:
            bias = 0

        return weight_ih, weight_hh, bias

    def forward_(self, x, hidden_states, sharpen_loss):
        if self.loss_to_sharpen is not None:
            sharpen_loss = self.loss_to_sharpen
            weight_ih, weight_hh, bias = self.sharpen_posterior(
                loss=sharpen_loss, input_shape=x.shape
            )
        elif sharpen_loss is not None:
            sharpen_loss = sharpen_loss
            weight_ih, weight_hh, bias = self.sharpen_posterior(
                loss=sharpen_loss, input_shape=x.shape
            )

        else:
            weight_ih, weight_hh, bias = self.sample_weights()

        # Assumes x is of shape (batch, sequence, feature)
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        # if no hidden state, we are using zeros
        if hidden_states is None:
            h_t = torch.zeros(bs, self.out_features).to(x.device)
        else:
            h_t = hidden_states

        # simplifying our out features, and hidden seq list
        HS = self.out_features
        hidden_seq = []

        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            A_t = (
                x_t @ weight_ih[:, : HS * 2]
                + h_t @ weight_hh[:, : HS * 2]
                + bias[: HS * 2]
            )

            r_t, z_t = (torch.sigmoid(A_t[:, :HS]), torch.sigmoid(A_t[:, HS : HS * 2]))

            n_t = torch.tanh(
                x_t @ weight_ih[:, HS * 2 : HS * 3]
                + bias[HS * 2 : HS * 3]
                + r_t * (h_t @ weight_hh[:, HS * 3 : HS * 4] + bias[HS * 3 : HS * 4])
            )
            h_t = (1 - z_t) * n_t + z_t * h_t

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, h_t

    def forward_frozen(self, x, hidden_states):
        weight_ih, weight_hh, bias = self.get_frozen_weights()

        # Assumes x is of shape (batch, sequence, feature)
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        # if no hidden state, we are using zeros
        if hidden_states is None:
            h_t = torch.zeros(bs, self.out_features).to(x.device)
        else:
            h_t = hidden_states

        # simplifying our out features, and hidden seq list
        HS = self.out_features
        hidden_seq = []

        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            A_t = (
                x_t @ weight_ih[:, : HS * 2]
                + h_t @ weight_hh[:, : HS * 2]
                + bias[: HS * 2]
            )

            r_t, z_t = (torch.sigmoid(A_t[:, :HS]), torch.sigmoid(A_t[:, HS : HS * 2]))

            n_t = torch.tanh(
                x_t @ weight_ih[:, HS * 2 : HS * 3]
                + bias[HS * 2 : HS * 3]
                + r_t * (h_t @ weight_hh[:, HS * 3 : HS * 4] + bias[HS * 3 : HS * 4])
            )
            h_t = (1 - z_t) * n_t + z_t * h_t

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, h_t

    def forward(self, x, hidden_states=None, sharpen_loss=None):
        if self.freeze:
            return self.forward_frozen(x, hidden_states)

        if not self.sharpen:
            sharpen_loss = None

        return self.forward_(x, hidden_states, sharpen_loss)


class BayesianLinear(BayesianModule):
    """
    Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).
    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        neuron_types_in: int -> number of input type neurons
        neuron_types_out: int -> number of output type neurons
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """

    def __init__(
        self,
        in_features,
        out_features,
        neuron_types_in,
        neuron_types_out,
        WS=True,
        bias=True,
        prior_sigma_1=0.1,
        prior_sigma_2=0.4,
        prior_pi=1,
        posterior_mu_init=0,
        posterior_rho_init=-7.0,
        freeze=False,
        prior_dist=None,
    ):
        super().__init__()

        # our main parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        self.pnet_size = in_features * out_features
        self.gnet_size = neuron_types_in * neuron_types_out

        self.WS = WS
        # Variational weight parameters and sample
        if self.WS == False:
            self.weight_mu = nn.Parameter(
                torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1)
            )
            self.weight_rho = nn.Parameter(
                torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1)
            )
            self.weight_sampler = TrainableRandomDistribution(
                self.weight_mu, self.weight_rho
            )

            # Variational bias parameters and sample
            self.bias_mu = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_mu_init, 0.1)
            )
            self.bias_rho = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_rho_init, 0.1)
            )
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
        else:
            self.neuron_types_in = neuron_types_in
            self.neuron_types_out = neuron_types_out
            n_in_neurons_per_type = find_num_neuron_per_type(
                in_features, neuron_types_in
            )
            in_neuron_type_ids = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_in_neurons_per_type)], axis=0
            ).astype(np.int32)

            n_out_neurons_per_type = find_num_neuron_per_type(
                out_features, neuron_types_out
            )
            out_neuron_type_ids = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_out_neurons_per_type)], axis=0
            ).astype(np.int32)

            indices = get_indices(
                in_neuron_type_ids, in_features, out_neuron_type_ids, out_features
            )
            self.weight_mu_share = nn.Parameter(
                torch.Tensor(1, neuron_types_in, neuron_types_out).normal_(
                    posterior_mu_init, 0.1
                )
            )
            self.weight_rho_share = nn.Parameter(
                torch.Tensor(1, neuron_types_in, neuron_types_out).normal_(
                    posterior_rho_init, 0.1
                )
            )

            self.weight_sampler = TrainableRandomDistribution_weight_share(
                self.weight_mu_share, self.weight_rho_share, indices
            )

            self.bias_mu = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_mu_init, 0.1)
            )
            self.bias_rho = nn.Parameter(
                torch.Tensor(out_features).normal_(posterior_rho_init, 0.1)
            )
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.bias_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # Sample the weights and forward it

        # if the model is frozen, return frozen
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features), device=x.device)
            b_log_posterior = 0
            b_log_prior = 0

        # Get the complexity cost
        self.log_variational_posterior = (
            self.weight_sampler.log_posterior() + b_log_posterior
        )
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.linear(x, w, b)

    def forward_frozen(self, x):
        """
        Computes the feedforward operation with the expected value for weight and biases
        """
        if self.bias:
            return F.linear(x, self.weight_mu, self.bias_mu)
        else:
            return F.linear(x, self.weight_mu, torch.zeros(self.out_features))


class BayesianConv2d(BayesianModule):
    # Implements Bayesian Conv2d layer, by drawing them using Weight Uncertanity on Neural Networks algorithm
    """
    Bayesian Linear layer, implements a Convolution 2D layer as proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_channels: int -> incoming channels for the layer
        out_channels: int -> output channels for the layer
        kernel_size : tuple (int, int) -> size of the kernels for this convolution layer
        types: neuron types for parameters in order in_channels,out_channels,kernel_size_out,kernel_size_in
        groups : int -> number of groups on which the convolutions will happend
        padding : int -> size of padding (0 if no padding)
        dilation int -> dilation of the weights applied on the input tensor


        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        types,
        WS=True,
        groups=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        prior_sigma_1=0.1,
        prior_sigma_2=0.002,
        prior_pi=1,
        posterior_mu_init=0,
        posterior_rho_init=-6.0,
        freeze=False,
        prior_dist=None,
    ):
        super().__init__()

        # our main parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freeze = freeze
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        # parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist
        self.WS = WS

        if not self.WS:
            # our weights
            self.weight_mu = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(
                    posterior_mu_init, 0.1
                )
            )
            self.weight_rho = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size).normal_(
                    posterior_rho_init, 0.1
                )
            )
            self.weight_sampler = TrainableRandomDistribution(
                self.weight_mu, self.weight_rho
            )

        else:
            n_in_channels_types = types[0]
            n_out_channels_types = types[1]
            n_out_kernel_types = types[2]
            n_in_kernel_types = types[3]

            self.pnet_size = (
                out_channels * (in_channels // groups) * kernel_size[0] * kernel_size[1]
            )
            self.gnet_size = (
                n_out_channels_types
                * (n_in_channels_types // groups)
                * n_out_kernel_types
                * n_in_kernel_types
            )

            n_kernel_in_neurons_per_type = find_num_neuron_per_type(
                kernel_size[0], n_in_kernel_types
            )
            pre_unit_types = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_kernel_in_neurons_per_type)],
                axis=0,
            ).astype(np.int32)

            n_kernel_out_neurons_per_type = find_num_neuron_per_type(
                kernel_size[1], n_out_kernel_types
            )
            post_unit_types = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_kernel_out_neurons_per_type)],
                axis=0,
            ).astype(np.int32)

            n_in_channel_neurons_per_type = find_num_neuron_per_type(
                self.in_channels, n_in_channels_types
            )
            pre_channel_types = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_in_channel_neurons_per_type)],
                axis=0,
            ).astype(np.int32)

            n_kernel_out_neurons_per_type = find_num_neuron_per_type(
                self.out_channels, n_out_channels_types
            )
            post_channel_types = np.concatenate(
                [i * np.ones(n) for i, n in enumerate(n_kernel_out_neurons_per_type)],
                axis=0,
            ).astype(np.int32)

            indices = get_indices_CNN(
                [self.out_channels, self.in_channels, kernel_size[1], kernel_size[0]],
                [
                    post_channel_types,
                    pre_channel_types,
                    post_unit_types,
                    pre_unit_types,
                ],
            )
            self.weight_mu = nn.Parameter(
                torch.Tensor(
                    1,
                    n_out_channels_types,
                    n_in_channels_types // groups,
                    n_out_kernel_types,
                    n_in_kernel_types,
                ).normal_(posterior_mu_init, 0.1)
            )
            self.weight_rho = nn.Parameter(
                torch.Tensor(
                    1,
                    n_out_channels_types,
                    n_in_channels_types // groups,
                    n_out_kernel_types,
                    n_in_kernel_types,
                ).normal_(posterior_rho_init, 0.1)
            )
            self.weight_sampler = TrainableRandomDistribution_weight_share_CNN(
                self.weight_mu, self.weight_rho, indices
            )

        # our biases
        if self.bias:
            self.bias_mu = nn.Parameter(
                torch.Tensor(out_channels).normal_(posterior_mu_init, 0.1)
            )
            self.bias_rho = nn.Parameter(
                torch.Tensor(out_channels).normal_(posterior_rho_init, 0.1)
            )
            self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)
            self.bias_prior_dist = PriorWeightDistribution(
                self.prior_pi,
                self.prior_sigma_1,
                self.prior_sigma_2,
                dist=self.prior_dist,
            )
        else:
            self.register_buffer("bias_zero", torch.zeros((self.out_channels)))

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # Forward with uncertain weights, fills bias with zeros if layer has no bias
        # Also calculates the complecity cost for this sampling
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = self.bias_zero
            b_log_posterior = 0
            b_log_prior = 0

        self.log_variational_posterior = (
            self.weight_sampler.log_posterior() + b_log_posterior
        )
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.conv2d(
            input=x,
            weight=w,
            bias=b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward_frozen(self, x):
        # Computes the feedforward operation with the expected value for weight and biases (frozen-like)

        if self.bias:
            bias = self.bias_mu
            assert (
                bias is self.bias_mu
            ), "The bias inputed should be this layer parameter, not a clone."
        else:
            bias = self.bias_zero

        return F.conv2d(
            input=x,
            weight=self.weight_mu,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
