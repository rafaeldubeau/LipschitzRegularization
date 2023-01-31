
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from collections import OrderedDict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LipschitzLayer(nn.Linear):
    """
    A fully-connected layer with learnable Lipschitz regularization

    See: https://www.dgp.toronto.edu/~hsuehtil/pdf/lipmlp.pdf


    Attributes:
        c: learned Lipschitz constant for this layer

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        """
        Initializes linear layer and lipschitz constant equal to 1


        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``

        """
        super().__init__(in_features, out_features, bias, device, dtype)

        self.c = Parameter(torch.tensor(1, dtype=torch.float))

        self._eval_weight = self.weight
        self._update_eval_weight = True
    
    def _normalization(self, W):
        """
        Normalizes W s.t. the sum of the absolute values of each row is less than softplus(self.c)

        If the sum of the absolute values of a row is already less than self.c, then
        the row is not changed.


        Args:
            W: the matrix to be normalized
        
        Returns:
            The normalized matrix

        """
        absrowsum = torch.sum(torch.abs(W), dim=1)
        softplus_c = torch.log(1 + torch.exp(self.c))
        scale = torch.clamp(softplus_c/absrowsum, max=1.0)
        return W * scale[:, None]

    def forward(self, x):
        """
        Evaluates the layer on the input x

        Args:
            x: the input to the layer
        
        Returns:
            The output of layer evaluated on the input x

        """
        if self.training:
            if not self._update_eval_weight:
                self._update_eval_weight = True
            return F.linear(x, self._normalization(self.weight), self.bias)

        elif not self.training:
            if self._update_eval_weight:
                # Fix weights during evaluation to reduce unnecessary computation
                self._eval_weight = Parameter(self._normalization(self.weight))
                self._update_eval_weight = False
            return F.linear(x, self._eval_weight, self.bias)


class DeepSDF(nn.Module):
    """
    Neural network architecture for learning SDFs of multiple objects with optional
    lipschitz regularization for smooth interpolation between SDFs

    See: https://www.dgp.toronto.edu/~hsuehtil/pdf/lipmlp.pdf

    """

    def __init__(self, input_dim: int = 2, latent_dim: int = 0, num_hidden: int = 5, hidden_dim: int = 256, is_lipschitz: bool = False):
        """
        Initializes neural network

        The size of the input layer is (input_dim + latent_dim) x hidden_dim

        Args:
            input_dim: dimension of input points
            latent_dim: dimension of latent codes
            num_hidden: number of hidden layers
            hidden_dim: size of each hidden layer (i.e. each hidden layer is hidden_dim x hidden_dim)
            is_lipschitz: determines whether or not the network is initialized with Lipschitz regularized layers

        """
        super(DeepSDF, self).__init__()

        if is_lipschitz:
            layer = LipschitzLayer
        else:
            layer = nn.Linear

        layer_dict = OrderedDict()
        layer_dict["input"] = layer(input_dim + latent_dim, hidden_dim)
        layer_dict["input_activation"] = nn.Tanh()
        for i in range(num_hidden):
            layer_dict[f"hidden{i}"] = layer(hidden_dim, hidden_dim)
            layer_dict[f"hidden{i}_activation"] = nn.Tanh()
        layer_dict["output"] = layer(hidden_dim, 1)

        self._layers = nn.Sequential(layer_dict)
    
    def forward(self, x):
        """
        Evaluates the neural network on the input x
        Latent codes should be appended to the input before being passed to the network

        Args:
            x: the input to the network
        
        Returns:
            The learned signed-distance value at a certain point for an object with the given
            latent code

        """
        return self._layers(x)
    
    def get_lipschitz_bound(self):
        """
        Computes an upper-bound of the lipschitz value of the neural network

        Returns:
            An upper-bound of the network's lipschitz value
        
        """
        prod = None
        for m in self._layers.modules():
            if isinstance(m, LipschitzLayer):
                if prod is None:
                    prod = m.c
                else:
                    prod = prod * torch.log(1 + torch.exp(m.c))
        if prod < 0:
            print(prod)
        return prod
        


def train_loop(model: DeepSDF, data: DataLoader, optimizer: Optimizer, loss_fn):
    """
    Trains a given model for one epoch

    Args:
        model: the model being trained
        data: a DataLoader containing the training data
        optimizer: the optimizer training the model's parameters
        loss_fn: a function to compute the loss to be minimized

    Returns:
        Nothing
    
    """
    if not model.training:
        model.train()

    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>7d}/{len(data.dataset):>7d}]")


def eval_loop(model: nn.Module, data: DataLoader, loss_fn):
    if model.training:
        model.eval()

    test_loss = 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y, model).item()

    test_loss /= len(data)
    print(f"Test Loss: {test_loss:>8f}\n")