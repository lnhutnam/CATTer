import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.torch_utils import gauss_ker, poly_ker2, poly_ker3, poly_ker5


class ISRLU(nn.Module):
    # https://openreview.net/pdf?id=HkMCybx0-
    @staticmethod
    def forward(x, alpha):
        if x >= 0:
            return x
        return x*(1/np.sqrt(1+alpha*x**2))


class Linear(nn.Module):
    @staticmethod
    def forward(x):
        return x


class Hardswish(nn.Module):
    # Hard-SiLU activation
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for TorchScript and CoreML
        # for TorchScript, CoreML and ONNX
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class MemoryEfficientMish(nn.Module):
    # Mish activation memory-efficient
    class F(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


class SMHT(nn.Module):
    # Soboleva Modified Hyperbolic Tangent function.
    @staticmethod
    def forward(x, params: np.ndarray = np.array([0.2, 0.4, 0.6, 0.8])):
        """
        Implements the Soboleva Modified Hyperbolic Tangent function

        Parameters:
            vector (ndarray): A vector that consists of numeric values
            params (ndarray): A vector that consists of parameters
                a_value (float): parameter a of the equation
                b_value (float): parameter b of the equation
                c_value (float): parameter c of the equation
                d_value (float): parameter d of the equation

        Returns:
            vector (ndarray): Input array after applying SMHT function

        >>> vector = np.array([5.4, -2.4, 6.3, -5.23, 3.27, 0.56])
        >>> params = np.array([0.2, 0.4, 0.6, 0.8])
        >>> soboleva_modified_hyperbolic_tangent(vector, params)
        array([ 0.11075085, -0.28236685,  0.07861169, -0.1180085 ,  0.22999056,
                0.1566043 ])
        """
        a_value: float
        b_value: float
        c_value: float
        d_value: float
        a_value, b_value, c_value, d_value = params

        # Separate the numerator and denominator for simplicity
        # Calculate the numerator and denominator element-wise
        numerator = torch.exp(a_value * x) - torch.exp(-b_value * x)
        denominator = torch.exp(c_value * x) + torch.exp(-d_value * x)

        # Calculate and return the final result element-wise
        return numerator / denominator


class SquarePlus(nn.Module):
    # Squareplus Activation Function
    @staticmethod
    def forward(x, beta: float = 2):
        """
        Implements the SquarePlus activation function.

        Parameters:
            vector (np.ndarray): The input array for the SquarePlus activation.
            beta (float): size of the curved region

        Returns:
            np.ndarray: The input array after applying the SquarePlus activation.

        Formula: f(x) = ( x + sqrt(x^2 + b) ) / 2

        Examples:
        >>> squareplus(np.array([2.3, 0.6, -2, -3.8]), beta=2)
        array([2.5       , 1.06811457, 0.22474487, 0.12731349])

        >>> squareplus(np.array([-9.2, -0.3, 0.45, -4.56]), beta=3)
        array([0.0808119 , 0.72891979, 1.11977651, 0.15893419])
        """
        return (x + torch.sqrt(x**2 + beta)) / 2


def Acts(f: str):
    function_dict = {
        "Linear": Linear(),
        "Logistic": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "SoftPlus": nn.Softplus(),
        "GELU": nn.GELU(),
        "SiLU": nn.SiLU(),
        "Hardswish": Hardswish(),
        "Mish": nn.Mish(),
        "MemoryEfficientMish": MemoryEfficientMish(),
        "SELU": nn.SELU(),
        "PReLU": nn.PReLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "SMHT": SMHT(),
        "SquarePlus": SquarePlus(),
        "GaussKer": gauss_ker,
        "PolyKer2": poly_ker2,
        "PolyKer3": poly_ker3,
        "PolyKer5": poly_ker5,
    }

    return function_dict[f]
