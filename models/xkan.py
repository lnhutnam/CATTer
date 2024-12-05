import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from typing import Union, Optional, Set, Tuple, List

##################################################################################
# B-Spline KAN
##################################################################################
# code modified from https://github.com/Blealtan/efficient-kan
# Basis function: B-Spline


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1,
                               self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(
            splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] +
                        2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + \
            (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1,
                               device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1,
                               device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(
            self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class BSpline_KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(BSpline_KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(
                regularize_activation, regularize_entropy)
            for layer in self.layers
        )


##################################################################################
# Bessel KAN
##################################################################################
class BesselKANLayer(nn.Module):
    # Kolmogorov-Arnold Networks but using Bessel polynomials instead of splines coefficients
    # Refs: https://github.com/Boris-73-TA/OrthogPolyKANs
    def __init__(self, input_dims: int, output_dims: int, degree: int = 7):
        super(BesselKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree

        # Initialize Bessel polynomial coefficients
        self.bessel_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, self.degree + 1))
        nn.init.normal_(self.bessel_coeffs, mean=0.0,
                        std=1/(self.input_dims * (self.degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dims)  # Reshape x to (batch_size, input_dim)
        # Normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # Initialize Bessel polynomial tensors
        bessel = torch.ones(x.shape[0], self.input_dims,
                            self.degree + 1, device=x.device)
        if self.degree > 0:
            bessel[:, :, 1] = x + 1  # y1(x) = x + 1
        for i in range(2, self.degree + 1):
            bessel[:, :, i] = (2 * i - 1) * x * bessel[:, :,
                                                       i - 1].clone() + bessel[:, :, i - 2].clone()

        # Bessel interpolation using einsum for batched matrix-vector multiplication
        # shape = (batch_size, output_dim)
        y = torch.einsum('bid,iod->bo', bessel, self.bessel_coeffs)
        y = y.view(-1, self.output_dims)
        return y


class BesselKANLayerWithNorm(nn.Module):
    # To avoid gradient vanishing caused by tanh
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(BesselKANLayerWithNorm, self).__init__()
        self.layer = BesselKANLayer(
            input_dims=input_dims, output_dims=output_dims, degree=degree)
        # To avoid gradient vanishing caused by tanh
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x


class Bessel_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 3,
        grid_size: int = 8,  # placeholder
        spline_order=0.  # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            BesselKANLayerWithNorm(
                input_dims=in_dims,
                output_dims=out_dims,
                degree=degree,
            ) for in_dims, out_dims in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Chebyshev KAN
##################################################################################
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(ChebyKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(
            self.input_dims, self.output_dims, self.degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 /
                        (self.input_dims * (self.degree + 1)))

    def forward(self, x):
        # shape = (batch_size, input_dims)
        x = torch.reshape(x, (-1, self.input_dims))
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Chebyshev polynomial tensors
        cheby = torch.ones(x.shape[0], self.input_dims,
                           self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - \
                cheby[:, :, i - 2].clone()
        # Compute the Chebyshev interpolation
        # shape = (batch_size, output_dims)
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)
        y = y.view(-1, self.output_dims)
        return y


class FasterChebyKANLayer(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(FasterChebyKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0,
                        std=1 / (self.input_dims * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, output_dims)
        y = y.view(-1, self.output_dims)
        return y


class ChebyKANLayerWithNorm(nn.Module):
    # To avoid gradient vanishing caused by tanh
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(ChebyKANLayerWithNorm, self).__init__()
        self.layer = FasterChebyKANLayer(
            input_dims=input_dims, output_dims=output_dims, degree=degree)
        # To avoid gradient vanishing caused by tanh
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x


class Chebyshev_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 4,
        grid_size: int = 8,  # placeholder
        spline_order=0.  # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ChebyKANLayerWithNorm(
                input_dims=in_dims,
                output_dims=out_dims,
                degree=degree,
            ) for in_dims, out_dims in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Fully-Connected Network KAN
##################################################################################
def heaviside_theta(x, mu, r):
    """Heaviside theta function with parameters mu and r.

    Args:
        x (torch.Tensor): Input tensor.
        mu (float): Center of the function.
        r (float): Width of the function.

    Returns:
        torch.Tensor: Output tensor.
    """
    x = x - mu
    return (torch.clamp(x + r, 0, r) - torch.clamp(x, 0, r)) / r


def _linear_interpolation(x, X, Y):
    """Linear interpolation function.

    Note: This function is used to apply the linear interpolation to one element of the input tensor.
    For vectorized operations, use the linear_interpolation function.

    Args:
        x (torch.Tensor): Input tensor.
        X (torch.Tensor): X values.
        Y (torch.Tensor): Y values.

    Returns:
        torch.Tensor: Output tensor.
    """
    mu = X
    r = X[1] - X[0]
    F = torch.vmap(heaviside_theta, in_dims=(None, 0, None))
    y = F(x, mu, r).reshape(-1) * Y
    return y.sum()


def linear_interpolation(x, X, Y):
    """Linear interpolation function.

    Args:
        x (torch.Tensor): Input tensor.
        X (torch.Tensor): X values.
        Y (torch.Tensor): Y values.

    Returns:
        torch.Tensor: Output tensor.
    """
    shape = x.shape
    x = x.reshape(-1)
    return torch.vmap(_linear_interpolation, in_dims=(-1, None, None), out_dims=-1)(x, X, Y).reshape(shape)


def phi(x, w1, w2, b1, b2, n_sin):
    """
    phi function that integrates sinusoidal embeddings with MLP layers.

    Args:
        x (torch.Tensor): Input tensor.
        w1 (torch.Tensor): Weight matrix for the first linear transformation.
        w2 (torch.Tensor): Weight matrix for the second linear transformation.
        b1 (torch.Tensor): Bias vector for the first linear transformation.
        b2 (torch.Tensor): Bias vector for the second linear transformation.
        n_sin (int): Number of sinusoidal functions to generate.

    Returns:
        torch.Tensor: Transformed tensor.
    """
    omega = (2 ** torch.arange(0, n_sin, device=x.device)
             ).float().reshape(-1, 1)
    omega_x = F.linear(x, omega, bias=None)
    x = torch.cat([x, torch.sin(omega_x), torch.cos(omega_x)], dim=-1)

    x = F.linear(x, w1, bias=b1)
    x = F.silu(x)
    x = F.linear(x, w2, bias=b2)
    return x


class FCNKANLayer(nn.Module):
    """
    A layer in a Kolmogorov-Arnold Networks (KAN).

    Attributes:
        input_dims (int): Dimensionality of the input.
        output_dims (int): Dimensionality of the output.
        fcn_hidden (int): Number of hidden units in the feature transformation.
        fcn_n_sin (torch.tensor): Number of sinusoidal functions to be used in phi.
    """

    def __init__(self, input_dims, output_dims, fcn_hidden=32, fcn_n_sin=3):
        """
        Initializes the KANLayer with specified dimensions and sinusoidal function count.

        Args:
            input_dims (int): Dimension of the input.
            output_dims (int): Dimension of the output.
            fcn_hidden (int): Number of hidden neurons in the for the learned non-linear transformation.
            fcn_n_sin (int): Number of sinusoidal embedding frequencies.
        """
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(
            input_dims, output_dims, fcn_hidden, 1+fcn_n_sin*2))
        self.W2 = nn.Parameter(torch.randn(
            input_dims, output_dims, 1, fcn_hidden))
        self.B1 = nn.Parameter(torch.randn(
            input_dims, output_dims, fcn_hidden))
        self.B2 = nn.Parameter(torch.randn(input_dims, output_dims, 1))

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fcn_hidden = fcn_hidden
        self.fcn_n_sin = torch.tensor(fcn_n_sin).long()

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)

        # apply zero bias
        nn.init.zeros_(self.B1)
        nn.init.zeros_(self.B2)

    def map(self, x):
        """
        Maps input tensor x through phi function in a vectorized manner.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after mapping through phi.
        """
        F = torch.vmap(
            # take input_dims out, -> input_dims x (dim_out, *)(1)
            # take dim_out out, -> dim_out x (*)
            torch.vmap(phi, (None, 0, 0, 0, 0, None), 0),
            (0, 0, 0, 0, 0, None), 0
        )
        return F(x.unsqueeze(-1), self.W1, self.W2, self.B1, self.B2, self.fcn_n_sin).squeeze(-1)

    def forward(self, x):
        """
        Forward pass of the KANLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Summed output after mapping each dimensions through phi.
        """
        device = x.device
        self.W1 = self.W1.to(device)
        self.W2 = self.W2.to(device)
        self.B1 = self.B1.to(device)
        self.B2 = self.B2.to(device)
        self.fcn_n_sin = self.fcn_n_sin.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        batch, input_dims = x.shape
        assert input_dims == self.input_dims

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x)  # [batch, input_dims, dim_out]

        return phis.sum(dim=1)

    def take_function(self, i, j):
        """
        Returns a phi function specific to the (i, j)-th elements of parameters.

        Args:
            i (int): Row index in parameter tensors.
            j (int): Column index in parameter tensors.

        Returns:
            function: A function that computes phi for specific parameters.
        """
        def activation(x):
            return phi(x, self.W1[i, j], self.W2[i, j], self.B1[i, j], self.B2[i, j], self.fcn_n_sin)
        return activation


class FCNKANInterpoLayer(nn.Module):
    """
    A layer in a Kolmogorov-Arnold Networks (KAN).

    Attributes:
        input_dims (int): Dimensionality of the input.
        output_dims (int): Dimensionality of the output.
        num_x (int): Number of x values to interpolate.
        x_min (float): Minimum x value.
    """

    def __init__(self, input_dims, output_dims, num_x=64, x_min=-2, x_max=2):
        """
        Initializes the KANLayer with specified dimensions and sinusoidal function count.

        Args:
            input_dims (int): Dimension of the input.
            output_dims (int): Dimension of the output.
            num_x (int): Number of x values to interpolate.
            x_min (float): Minimum x value.
        """
        super().__init__()
        # self.X = nn.Parameter(torch.randn(input_dims, dim_out, num_x)
        self.X = torch.linspace(x_min, x_max, num_x)
        self.Y = nn.Parameter(torch.randn(input_dims, output_dims, num_x))

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.Y)

    def map(self, x):
        """
        Maps input tensor x through phi function in a vectorized manner.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after mapping through phi.
        """
        F = torch.vmap(
            # take input_dims out, -> input_dims x (dim_out, *)(1)
            # take dim_out out, -> dim_out x (*)
            torch.vmap(linear_interpolation, (None, None, 0), 0),
            (0, None, 0), 0
        )
        return F(x.unsqueeze(-1), self.X, self.Y).squeeze(-1)

    def forward(self, x):
        """
        Forward pass of the KANLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Summed output after mapping each dimensions through phi.
        """
        device = x.device
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        batch, input_dims = x.shape
        assert input_dims == self.input_dims

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x)  # [batch, input_dims, dim_out]

        return phis.sum(dim=1)

    def take_function(self, i, j):
        """
        Returns a phi function specific to the (i, j)-th elements of parameters.

        Args:
            i (int): Row index in parameter tensors.
            j (int): Column index in parameter tensors.

        Returns:
            function: A function that computes phi for specific parameters.
        """
        def activation(x):
            return linear_interpolation(x, self.X, self.Y[i, j])
        return activation


def smooth_penalty(model):
    p = 0
    if isinstance(model, FCNKANInterpoLayer):
        dx = model.X[1] - model.X[0]
        grad = model.Y[:, :, 1:] - model.Y[:, :, :-1]
        # grad = grad[:, :, 1:] - grad[:, :, :-1]
        return torch.norm(grad, 2) / dx

    for layer in model:
        if isinstance(layer, FCNKANInterpoLayer):
            dx = layer.X[1] - layer.X[0]
            grad = layer.Y[:, :, 1:] - layer.Y[:, :, :-1]
            # grad = grad[:, :, 1:] - grad[:, :, :-1]
            p += torch.norm(grad, 2) / dx
    return p


class FCN_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8,  # placeholder
        spline_order: int = 0,  # placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FCNKANLayer(
                input_dims=in_dims,
                output_dims=out_dims,
                fcn_hidden=1,  # default ?
                fcn_n_sin=1  # default ?
            ) for in_dims, out_dims in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FCN_InterpoKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8,  # placeholder
        spline_order: int = 0,  # placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FCNKANInterpoLayer(
                input_dims=in_dims,
                output_dims=out_dims,
                num_x=8,  # default
                x_min=-2,  # default
                x_max=2   # default
            ) for in_dims, out_dims in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Fibonacci KAN
##################################################################################
class FibonacciKANLayer(nn.Module):
    # Kolmogorov-Arnold Networks but using Fibonacci polynomials instead of splines coefficients
    # Refs: https://github.com/Boris-73-TA/OrthogPolyKANs
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(FibonacciKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree

        # Initialize coefficients for the Fibonacci polynomials
        self.fib_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, self.degree + 1))
        nn.init.normal_(self.fib_coeffs, mean=0.0,
                        std=1 / (self.input_dims * (self.degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dims)  # Reshape to (batch_size, input_dim)
        # Normalize input x to [-1, 1] for stability in polynomial calculation
        x = torch.tanh(x)

        # Initialize Fibonacci polynomial tensors
        fib = torch.zeros(x.size(0), self.input_dims,
                          self.degree + 1, device=x.device)
        fib[:, :, 0] = 0  # F_0(x) = 0
        if self.degree > 0:
            fib[:, :, 1] = 1  # F_1(x) = 1

        for i in range(2, self.degree + 1):
            # Compute Fibonacci polynomials using the recurrence relation
            fib[:, :, i] = x * fib[:, :, i - 1].clone() + \
                fib[:, :, i - 2].clone()

        # Normalize the polynomial outputs to prevent runaway values
        # fib = torch.tanh(fib)

        # Compute the Fibonacci interpolation
        # shape = (batch_size, output_dim)
        y = torch.einsum('bid,iod->bo', fib, self.fib_coeffs)
        y = y.view(-1, self.output_dims)
        return y


class FibonacciKANLayerWithNorm(nn.Module):
    # To avoid gradient vanishing caused by tanh
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(FibonacciKANLayerWithNorm, self).__init__()
        self.layer = FibonacciKANLayer(
            input_dims=input_dims, output_dims=output_dims, degree=degree)
        # To avoid gradient vanishing caused by tanh
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x


class Fibonacci_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 4,
        grid_size: int = 8,  # placeholder
        spline_order=0.  # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FibonacciKANLayerWithNorm(
                input_dims=in_dim,
                output_dims=out_dim,
                degree=degree,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Fourier KAN
##################################################################################
# This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
# It should be easier to optimize as fourier are more dense than spline (global vs local)
# Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
# The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
# Avoiding the issues of going out of grid
class FourierKANLayer(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, grid_size: int, add_bias=True):
        super(FourierKANLayer, self).__init__()
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.input_dims = input_dims
        self.output_dims = output_dims

        # The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        # then each coordinates of the output is of unit variance
        # independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter(torch.randn(2, self.output_dims, self.input_dims, self.grid_size) /
                                                (np.sqrt(input_dims) * np.sqrt(self.grid_size)))
        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, self.output_dims))

    # x.shape ( ... , indim )
    # out.shape ( ..., output_dims)
    def forward(self, x):
        xshp = x.shape
        out_shape = xshp[0:-1]+(self.output_dims,)
        x = torch.reshape(x, (-1, self.input_dims))
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.grid_size+1,
                          device=x.device), (1, 1, 1, self.gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        # This should be fused to avoid materializing memory
        c = torch.cos(k*xrshp)
        s = torch.sin(k*xrshp)
        # We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        y = torch.sum(c*self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s*self.fouriercoeffs[1:2], (-2, -1))
        if (self.addbias):
            y += self.bias
        # End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape(y, out_shape)
        return y


class Fourier_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8,
        spline_order: int = 0,  # placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FourierKANLayer(
                input_dims=in_dim,
                output_dims=out_dim,
                grid_size=grid_size,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Gegenbauer KAN
##################################################################################
class GegenbauerKANLayer(nn.Module):
    # Kolmogorov-Arnold Networks but using Gegenbauer polynomials instead of splines coefficients
    # Refs: https://github.com/Boris-73-TA/OrthogPolyKANs
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3, alpha_param=1):
        super(GegenbauerKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree
        self.alpha_param = alpha_param

        # Initialize Gegenbauer polynomial coefficients
        self.gegenbauer_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, self.degree + 1))
        nn.init.normal_(self.gegenbauer_coeffs, mean=0.0,
                        std=1/(self.input_dims * (self.degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dims)  # Reshape to (batch_size, input_dim)
        x = torch.tanh(x)  # Normalize x to [-1, 1]

        gegenbauer = torch.ones(
            x.shape[0], self.input_dims, self.degree + 1, device=x.device)
        if self.degree > 0:
            gegenbauer[:, :, 1] = 2 * self.alpha_param * \
                x  # C_1^alpha(x) = 2*alpha*x

        for n in range(1, self.degree):
            term1 = 2 * (n + self.alpha_param) * x * \
                gegenbauer[:, :, n].clone()
            term2 = (n + 2 * self.alpha_param - 1) * \
                gegenbauer[:, :, n - 1].clone()
            # Apply the recurrence relation
            gegenbauer[:, :, n + 1] = (term1 - term2) / (n + 1)

        y = torch.einsum('bid,iod->bo', gegenbauer, self.gegenbauer_coeffs)
        return y.view(-1, self.output_dims)


class GegenbauerKANLayerWithNorm(nn.Module):
    # To avoid gradient vanishing caused by tanh
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3, alpha_param = 1):
        super(GegenbauerKANLayerWithNorm, self).__init__()
        self.layer = GegenbauerKANLayer(
            input_dims=input_dims, output_dims=output_dims, degree=degree, alpha_param=alpha_param)
        # To avoid gradient vanishing caused by tanh
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x


class Gegenbauer_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 4,
        alpha=3.,
        grid_size: int = 8,  # placeholder
        spline_order=0.  # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            GegenbauerKANLayerWithNorm(
                input_dims=in_dim,
                output_dims=out_dim,
                degree=degree,
                alpha_param=alpha
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Gaussian Radial Basis Functions KAN
##################################################################################
class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)

    def forward(self, x):
        return torch.exp(-(x[..., None] - self.grid) ** 2)


class GRBFKANLayer(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        base_activation=nn.SiLU,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dims)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.base_activation = base_activation()
        self.base_linear = nn.Linear(input_dims, output_dims)
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(output_dims, input_dims, num_grids)
        )
        nn.init.trunc_normal_(self.spline_weight, mean=0.0,
                              std=spline_weight_init_scale)

    def forward(self, x):
        base = self.base_linear(self.base_activation(x))
        spline_basis = self.rbf(self.layernorm(x))
        spline = torch.einsum(
            "...in,oin->...o", spline_basis, self.spline_weight
        )
        return base + spline


class GRBF_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        grid_size: int = 8,
        base_activation=nn.SiLU,
        spline_weight_init_scale: float = 0.1,
        spline_order=0.
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            GRBFKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=grid_size,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Hermite KAN
##################################################################################
class HermiteKANLayer(nn.Module):
    # Kolmogorov-Arnold Networks but using Hermite polynomials instead of splines coefficients
    # Refs: https://github.com/Boris-73-TA/OrthogPolyKANs
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(HermiteKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree

        # Initialize Hermite polynomial coefficients
        self.hermite_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, self.degree + 1))
        nn.init.normal_(self.hermite_coeffs, mean=0.0,
                        std=1/(self.input_dims * (self.degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        hermite = torch.ones(
            x.shape[0], self.input_dims, self.degree + 1, device=x.device)
        if self.degree > 0:
            hermite[:, :, 1] = 2 * x
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = 2 * x * hermite[:, :, i -
                                               1].clone() - 2 * (i - 1) * hermite[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', hermite, self.hermite_coeffs)
        y = y.view(-1, self.output_dims)
        return y


class HermiteKANLayerWithNorm(nn.Module):
    # To avoid gradient vanishing caused by tanh
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(HermiteKANLayerWithNorm, self).__init__()
        self.layer = HermiteKANLayer(
            input_dims=input_dims, output_dims=output_dims, degree=degree)
        # To avoid gradient vanishing caused by tanh
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x


class Hermite_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 4,
        grid_size: int = 8,  # placeholder
        spline_order=0.  # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            HermiteKANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Jacobian KAN
##################################################################################
class JacobiKANLayer(nn.Module):
    # Kolmogorov-Arnold Networks but using Jacobian polynomials instead of splines coefficients
    # Refs: https://github.com/SpaceLearner/JacobiKAN
    def __init__(self, input_dims: int, output_dims: int, degree=3, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, self.degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0,
                        std=1/(self.input_dims * (self.degree + 1)))

    def forward(self, x):
        # shape = (batch_size, inputdim)
        x = torch.reshape(x, (-1, self.input_dims))
        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.input_dims,
                            self.degree + 1, device=x.device)
        # degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
        if self.degree > 0:
            jacobi[:, :, 1] = ((self.a-self.b) + (self.a+self.b+2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2*i+self.a+self.b)*(2*i+self.a+self.b-1) / \
                (2*i*(i+self.a+self.b))
            theta_k1 = (2*i+self.a+self.b-1)*(self.a*self.a-self.b *
                                              self.b) / (2*i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            theta_k2 = (i+self.a-1)*(i+self.b-1)*(2*i+self.a+self.b) / \
                (i*(i+self.a+self.b)*(2*i+self.a+self.b-2))
            # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :,
                                                                i - 1].clone() - theta_k2 * jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        # shape = (batch_size, outdim)
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)
        y = y.view(-1, self.outdim)
        return y


class JacobiKANLayerWithNorm(nn.Module):
    # To avoid gradient vanishing caused by tanh
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3, a: float = 1.0, b: float = 1.0):
        super(JacobiKANLayerWithNorm, self).__init__()
        self.layer = JacobiKANLayer(input_dims, output_dims, degree, a, b)
        # To avoid gradient vanishing caused by tanh
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x


class Jacobi_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 3,
        a=1.0,
        b=1.0,
        grid_size: int = 8,  # placeholder
        spline_order=0.  # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            JacobiKANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
                a=a,
                b=b
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Laguerre KAN
##################################################################################
class LaguerreKANLayer(nn.Module):
    # Kolmogorov-Arnold Networks but using Laguerre polynomials instead of splines coefficients
    # Refs: https://github.com/Boris-73-TA/OrthogPolyKANs
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3, alpha: float = 1):
        super(LaguerreKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree
        self.alpha = alpha  # Alpha parameter for generalized Laguerre polynomials

        # Initialize coefficients for the Laguerre polynomials
        self.laguerre_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, self.degree + 1))
        nn.init.normal_(self.laguerre_coeffs, mean=0.0,
                        std=1 / (self.input_dims * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dims)  # Reshape to (batch_size, input_dim)
        # Normalize input x to [-1, 1] for stability in polynomial calculation
        x = torch.tanh(x)

        # Initialize Laguerre polynomial tensors
        laguerre = torch.zeros(x.size(0), self.input_dims,
                               self.degree + 1, device=x.device)
        laguerre[:, :, 0] = 1  # L_0^alpha(x) = 1
        if self.degree > 0:
            laguerre[:, :, 1] = 1 + self.alpha - \
                x  # L_1^alpha(x) = 1 + alpha - x

        for k in range(2, self.degree + 1):
            # Compute Laguerre polynomials using the generalized recurrence relation
            term1 = ((2 * (k-1) + 1 + self.alpha - x)
                     * laguerre[:, :, k - 1].clone())
            term2 = (k - 1 + self.alpha) * laguerre[:, :, k - 2].clone()
            laguerre[:, :, k] = (term1 - term2) / (k)

        # Normalize the polynomial outputs to prevent runaway values
        # laguerre = torch.tanh(laguerre)

        # Compute the Laguerre interpolation
        # shape = (batch_size, output_dim)
        y = torch.einsum('bid,iod->bo', laguerre, self.laguerre_coeffs)
        y = y.view(-1, self.output_dims)
        return y


class LaguerreKANLayerNorm(nn.Module):
    # To avoid gradient vanishing caused by tanh
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3, alpha = 1):
        super(LaguerreKANLayerNorm, self).__init__()
        self.layer = LaguerreKANLayer(
            input_dims=input_dims, output_dims=output_dims, degree=degree, alpha=alpha)
        # To avoid gradient vanishing caused by tanh
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x


class Laguerre_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 3,
        alpha: float = -0.5,
        grid_size: int = 8,  # placeholder
        spline_order=0.  # placeholder
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            LaguerreKANLayerNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
                alpha=alpha
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Legendre KAN
##################################################################################
class LegendreKANLayer(nn.Module):
    # Kolmogorov-Arnold Networks but using Legendre polynomials instead of splines coefficients
    # Refs: https://github.com/Boris-73-TA/OrthogPolyKANs
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(LegendreKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree
        self.legendre_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, self.degree + 1))
        nn.init.normal_(self.legendre_coeffs, mean=0.0,
                        std=1 / (self.input_dims * (self.degree + 1)))

    def forward(self, x):
        # shape = (batch_size, inputdim)
        x = torch.reshape(x, (-1, self.input_dims))
        # Normalize input to [-1, 1] for stability in Legendre polynomial calculation
        x = torch.tanh(x)

        # Initialize Legendre polynomial tensors
        legendre = torch.ones(
            x.shape[0], self.input_dims, self.degree + 1, device=x.device)
        legendre[:, :, 0] = 1  # P_0(x) = 1
        if self.degree > 0:
            legendre[:, :, 1] = x  # P_1(x) = x

        # Compute Legendre polynomials using the recurrence relation
        for n in range(2, self.degree + 1):
           # Recurrence relation without in-place operations
            legendre[:, :, n] = ((2 * (n-1) + 1) / (n)) * x * legendre[:,
                                                                       :, n-1].clone() - ((n-1) / (n)) * legendre[:, :, n-2].clone()

        # Compute output using matrix multiplication
        y = torch.einsum('bid,iod->bo', legendre, self.legendre_coeffs)
        y = y.view(-1, self.output_dims)
        return y


class LegendreKANLayerWithNorm(nn.Module):
    # To avoid gradient vanishing caused by tanh
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(LegendreKANLayerWithNorm, self).__init__()
        self.layer = LegendreKANLayer(
            input_dims=input_dims, output_dims=output_dims, degree=degree)
        # To avoid gradient vanishing caused by tanh
        self.layer_norm = nn.LayerNorm(output_dims)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x


class Legendre_kan(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 4,
        grid_size: int = 8,  # placeholder
        spline_order=0.  # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            LegendreKANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Lucas KAN
##################################################################################
class LucasKANLayer(nn.Module):
    # Kolmogorov-Arnold Networks but using Lucas polynomials instead of splines coefficients
    # Refs: https://github.com/Boris-73-TA/OrthogPolyKANs
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(LucasKANLayer, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.degree = degree

        # Initialize coefficients for the Lucas polynomials
        self.lucas_coeffs = nn.Parameter(
            torch.empty(self.input_dims, self.output_dims, self.degree + 1))
        nn.init.normal_(self.lucas_coeffs, mean=0.0,
                        std=1 / (self.input_dims * (self.degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dims)  # Reshape to (batch_size, input_dim)
        # Normalize input x to [-1, 1] for stability in polynomial calculation
        x = torch.tanh(x)

        # Initialize Lucas polynomial tensors
        lucas = torch.zeros(x.size(0), self.input_dims,
                            self.degree + 1, device=x.device)
        lucas[:, :, 0] = 2  # L_0(x) = 2
        if self.degree > 0:
            lucas[:, :, 1] = x  # L_1(x) = x

        for i in range(2, self.degree + 1):
            # Compute Lucas polynomials using the recurrence relation
            lucas[:, :, i] = x * lucas[:, :, i - 1].clone() + lucas[:, :,
                                                                    i - 2].clone()

        # Normalize the polynomial outputs to prevent runaway values
        # lucas = torch.tanh(lucas)

        # Compute the Lucas interpolation
        # shape = (batch_size, output_dim)
        y = torch.einsum('bid,iod->bo', lucas, self.lucas_coeffs)
        y = y.view(-1, self.output_dims)
        return y


class LucasKANLayerWithNorm(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, degree: int = 3):
        super(LucasKANLayerWithNorm, self).__init__()
        self.layer = LucasKANLayer(input_dims=input_dims, output_dims=output_dims, degree=degree)
        self.layer_norm = nn.LayerNorm(output_dims) # To avoid gradient vanishing caused by tanh

    def forward(self, x):
        x = self.layer(x)
        x = self.layer_norm(x)
        return x
    
class Lucas_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        degree: int = 3,
        grid_size: int = 8, # placeholder
        spline_order=0. # placehold
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            LucasKANLayerWithNorm(
                input_dim=in_dim,
                output_dim=out_dim,
                degree=degree,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
##################################################################################
# Nefacto_MLP KAN
##################################################################################
class Nefacto_MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        layers_hidden: List[int],
        grid_size: int = 8,  # placeholder
        spline_order: int = 0.,  # placeholder
        layer_width: int = 256,
        skip_connections: Optional[Tuple[int]] = (4,),
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_dim = layers_hidden[0]
        assert self.in_dim > 0
        self.out_dim = layers_hidden[-1]
        self.num_layers = len(layers_hidden)
        self.layer_width = layers_hidden[1]  # hidden_dim
        # self.layer_width = layer_width # hidden_dim
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(
            skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation

        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width +
                                  self.in_dim, self.layer_width))
                else:
                    layers.append(
                        nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, in_tensor):
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

##################################################################################
# RBF KAN
##################################################################################
# code modified from https://github.com/sidhu2690/RBF-KAN


class RBFLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_min=-2., grid_max=2., num_grids=8, spline_weight_init_scale=0.1):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.grid = nn.Parameter(torch.linspace(
            grid_min, grid_max, num_grids), requires_grad=False)
        self.spline_weight = nn.Parameter(torch.randn(
            in_features*num_grids, out_features)*spline_weight_init_scale)

    def forward(self, x):
        x = x.unsqueeze(-1)
        basis = torch.exp(-((x - self.grid) / ((self.grid_max -
                          self.grid_min) / (self.num_grids - 1))) ** 2)
        return basis.view(basis.size(0), -1).matmul(self.spline_weight)


class RBFKANLayer(nn.Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 grid_min: float = -2.,
                 grid_max: float = 2.,
                 num_grids: int = 8,
                 base_activation=nn.SiLU,
                 spline_weight_init_scale: float = 0.1,
                 use_base_update=True):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_base_update = use_base_update
        self.base_activation = base_activation()
        self.spline_weight_init_scale = spline_weight_init_scale
        self.rbf_linear = RBFLinear(
            input_dims, output_dims, grid_min, grid_max, num_grids, spline_weight_init_scale)
        self.base_linear = nn.Linear(
            input_dims, output_dims) if use_base_update else None

    def forward(self, x):
        ret = self.rbf_linear(x)
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class RBF_KAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        grid_size: int = 8,
        base_activation=nn.SiLU,
        use_base_update=True,
        spline_weight_init_scale: float = 0.1,
        spline_order=0.
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RBFKANLayer(
                input_dims=in_dim,
                output_dims=out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=grid_size,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# WAV KAN
##################################################################################
# code modified from https://github.com/zavareh1/Wav-KAN

class WavKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WavKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # Linear weights for combining outputs
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # not used; you may like to use it for wieghting base activation and adding it like Spl-KAN paper
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(
            torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(
            0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        # Implementation of different wavelet types
        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * \
                self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * \
                self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)

        elif self.wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * \
                self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2, torch.ones_like(v), torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * \
                self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(
                x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * \
                self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'bump':
            # Bump wavelet is only defined in the interval (-1, 1)
            # We apply a condition to restrict the computation to this interval
            inside_interval = (x_scaled > -1.0) & (x_scaled < 1.0)
            wavelet = torch.exp(-1.0 / (1 - x_scaled**2)) * \
                inside_interval.float()
            wavelet_weighted = wavelet * \
                self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        # You may like test the cases like Spl-KAN
        # wav_output = F.linear(wavelet_output, self.weight)
        # base_output = F.linear(self.base_activation(x), self.weight1)

        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output  # + base_output

        # Apply batch normalization
        return self.bn(combined_output)


class Mexican_Hat_KAN(nn.Module):
    def __init__(self, layers_hidden,
                 grid_size=5,  # placeholder
                 spline_order=0.,  # placeholder
                 wavelet_type='mexican_hat'):
        super(Mexican_Hat_KAN, self).__init__()
        super().__init__()
        self.layers = nn.ModuleList([
            WavKANLinear(
                in_features=in_dim,
                out_features=out_dim,
                wavelet_type=wavelet_type
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Morlet_KAN(nn.Module):
    def __init__(self, layers_hidden,
                 grid_size=5,  # placeholder
                 spline_order=0.,  # placeholder
                 wavelet_type='morlet'):
        super(Morlet_KAN, self).__init__()
        super().__init__()
        self.layers = nn.ModuleList([
            WavKANLinear(
                in_features=in_dim,
                out_features=out_dim,
                wavelet_type=wavelet_type
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Dog_KAN(nn.Module):
    def __init__(self, layers_hidden,
                 grid_size=5,  # placeholder
                 spline_order=0.,  # placeholder
                 wavelet_type='dog'):
        super(Dog_KAN, self).__init__()
        super().__init__()
        self.layers = nn.ModuleList([
            WavKANLinear(
                in_features=in_dim,
                out_features=out_dim,
                wavelet_type=wavelet_type
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Meyer_KAN(nn.Module):
    def __init__(self, layers_hidden,
                 grid_size=5,  # placeholder
                 spline_order=0.,  # placeholder
                 wavelet_type='meyer'):
        super(Meyer_KAN, self).__init__()
        super().__init__()
        self.layers = nn.ModuleList([
            WavKANLinear(
                in_features=in_dim,
                out_features=out_dim,
                wavelet_type=wavelet_type
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Shannon_KAN(nn.Module):
    def __init__(self, layers_hidden,
                 grid_size=5,  # placeholder
                 spline_order=0.,  # placeholder
                 wavelet_type='shannon'):
        super(Shannon_KAN, self).__init__()
        super().__init__()
        self.layers = nn.ModuleList([
            WavKANLinear(
                in_features=in_dim,
                out_features=out_dim,
                wavelet_type=wavelet_type
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Bump_KAN(nn.Module):
    def __init__(self, layers_hidden,
                 grid_size=5,  # placeholder
                 spline_order=0.,  # placeholder
                 wavelet_type='bump'):
        super(Bump_KAN, self).__init__()
        super().__init__()
        self.layers = nn.ModuleList([
            WavKANLinear(
                in_features=in_dim,
                out_features=out_dim,
                wavelet_type=wavelet_type
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


##################################################################################
# Conv KAN
##################################################################################
class ConvKAN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: torch.nn.Module = torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1, 1),
    ):
        """
        Convolutional layer with KAN kernels. A drop-in replacement for torch.nn.Conv2d.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple): Stride of the convolution. Default: 1
            padding (int or tuple): Padding added to both sides of the input. Default: 0
            dilation (int or tuple): Spacing between kernel elements. Default: 1
            groups (int): Number of blocked connections from input channels to output channels. Default: 1
            padding_mode (str): Padding mode. Default: 'zeros'
            bias (bool): Added for compatibility with torch.nn.Conv2d and does make any effect. Default: True
            grid_size (int): Number of grid points for the spline. Default: 5
            spline_order (int): Order of the spline. Default: 3
            scale_noise (float): Scale of the noise. Default: 0.1
            scale_base (float): Scale of the base. Default: 1.0
            scale_spline (float): Scale of the spline. Default: 1.0
            enable_standalone_scale_spline (bool): Enable standalone scale for the spline. Default: True
            base_activation (torch.nn.Module): Activation function for the base. Default: torch.nn.SiLU
            grid_eps (float): Epsilon for the grid
            grid_range (tuple): Range of the grid. Default: (-1, 1).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        self._in_dim = (
            (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1]
        )
        self._reversed_padding_repeated_twice = tuple(
            x for x in reversed(self.padding) for _ in range(2)
        )

        if not bias:
            # warn the user that bias is not used
            # warnings.warn("Bias is not used in ConvKAN layer", UserWarning)
            pass

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.kan_layer = KANLinear(
            self._in_dim,
            out_channels // groups,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
            groups=groups,
        )

    def forward(self, x):
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice,
                      mode=self.padding_mode)
            padding = (0, 0)  # Reset padding because we already applied it
        else:
            padding = self.padding

        x_unf = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=padding,
            stride=self.stride,
            dilation=self.dilation,
        )

        batch_size, channels_and_elem, n_patches = x_unf.shape

        # Ensuring group separation is maintained in the input
        x_unf = (
            x_unf.permute(0, 2, 1)  # [B, H_out * W_out, channels * elems]
            .reshape(
                batch_size * n_patches, self.groups, channels_and_elem // self.groups
            )  # [B * H_out * W_out, groups, out_channels // groups]
            .permute(1, 0, 2)
        )  # [groups, B * H_out * W_out, out_channels // groups]

        output = self.kan_layer(
            x_unf
        )  # [groups, B * H_out * W_out, out_channels // groups]
        output = (
            output.permute(1, 0, 2).reshape(
                batch_size, n_patches, -1).permute(0, 2, 1)
        )

        # Compute output dimensions
        output_height = (
            x.shape[2]
            + 2 * padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        output_width = (
            x.shape[3]
            + 2 * padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1

        # Reshape output to the expected output format
        output = output.view(
            x.shape[0],  # batch size
            self.out_channels,  # total output channels
            output_height,
            output_width,
        )

        return output


def _pair(x):
    if isinstance(x, (int, float)):
        return x, x
    return x
