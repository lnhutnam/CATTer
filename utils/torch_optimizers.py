"""PyTorch implementation of the Lion optimizer."""

import math
import torch
from torch.optim.optimizer import Optimizer
from torch import linalg as LA

version_higher = torch.__version__ >= "1.5.0"


class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        weight_decouple=False,
        fixed_decay=False,
        rectify=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(AdaBelief, self).__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print("Weight decoupling enabled in AdaBelief")
            if self.fixed_decay:
                print("Weight decay fixed")
        if self.rectify:
            print("Rectification enabled in AdaBelief")
        if amsgrad:
            print("AMS enabled in AdaBelief")

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                amsgrad = group["amsgrad"]

                # State initialization
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = (
                    torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if version_higher
                    else torch.zeros_like(p.data)
                )

                # Exponential moving average of squared gradient values
                state["exp_avg_var"] = (
                    torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if version_higher
                    else torch.zeros_like(p.data)
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_var"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaBelief does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["rho_inf"] = 2.0 / (1.0 - beta2) - 1.0
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_var"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_var"] = (
                            torch.zeros_like(
                                p.data, memory_format=torch.preserve_format
                            )
                            if version_higher
                            else torch.zeros_like(p.data)
                        )

                # get current state variable
                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        p.data.mul_(1.0 - group["weight_decay"])
                else:
                    if group["weight_decay"] != 0:
                        grad.add_(group["weight_decay"], p.data)

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    1 - beta2, grad_residual, grad_residual
                )

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (
                        max_exp_avg_var.add_(group["eps"]).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group["eps"])
                else:
                    denom = (
                        exp_avg_var.add_(group["eps"]).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group["eps"])

                if not self.rectify:
                    # Default update
                    step_size = group["lr"] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                else:  # Rectified update
                    # calculate rho_t
                    state["rho_t"] = state["rho_inf"] - 2 * state[
                        "step"
                    ] * beta2 ** state["step"] / (1.0 - beta2 ** state["step"])

                    if (
                        state["rho_t"] > 4
                    ):  # perform Adam style update if variance is small
                        rho_inf, rho_t = state["rho_inf"], state["rho_t"]
                        rt = (
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * rho_inf
                            / (rho_inf - 4.0)
                            / (rho_inf - 2.0)
                            / rho_t
                        )
                        rt = math.sqrt(rt)

                        step_size = rt * group["lr"] / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:  # perform SGD style update
                        p.data.add_(-group["lr"], exp_avg)

        return loss


class AdaBeliefNorm(Optimizer):
    r"""Implements AdaBeliefNorm algorithm. Modified from AdaBelief in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs
               WACV 2023
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        weight_decouple=False,
        fixed_decay=False,
        rectify=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(AdaBeliefNorm, self).__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print("Weight decoupling enabled in AdaBeliefNorm")
            if self.fixed_decay:
                print("Weight decay fixed")
        if self.rectify:
            print("Rectification enabled in AdaBeliefNorm")
        if amsgrad:
            print("AMS enabled in AdaBeliefNorm")

    def __setstate__(self, state):
        super(AdaBeliefNorm, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                amsgrad = group["amsgrad"]

                # State initialization
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = (
                    torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if version_higher
                    else torch.zeros_like(p.data)
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_var"] = (
                    torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if version_higher
                    else torch.zeros_like(p.data)
                )
                state["exp_grad_norm"] = 0

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_var"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in reversed(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdaBeliefNorm does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["rho_inf"] = 2.0 / (1.0 - beta2) - 1.0
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_var"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )
                    state["exp_grad_norm"] = 0
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_var"] = (
                            torch.zeros_like(
                                p.data, memory_format=torch.preserve_format
                            )
                            if version_higher
                            else torch.zeros_like(p.data)
                        )

                # get current state variable
                exp_avg, exp_avg_var, exp_grad_norm = (
                    state["exp_avg"],
                    state["exp_avg_var"],
                    state["exp_grad_norm"],
                )

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        p.data.mul_(1.0 - group["weight_decay"])
                else:
                    if group["weight_decay"] != 0:
                        grad.add_(group["weight_decay"], p.data)

                # Gradient Norm Correction
                grad_norm = LA.norm(grad)
                exp_grad_norm = 0.95 * exp_grad_norm + 0.05 * grad_norm
                if exp_grad_norm > grad_norm:
                    grad1 = grad * exp_grad_norm / grad_norm
                else:
                    grad1 = grad
                state["exp_grad_norm"] = exp_grad_norm.clone()

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad1)
                grad_residual = grad1 - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    1 - beta2, grad_residual, grad_residual
                )

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (
                        max_exp_avg_var.add_(group["eps"]).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group["eps"])
                else:
                    denom = (
                        exp_avg_var.add_(group["eps"]).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group["eps"])

                if not self.rectify:
                    # Default update
                    step_size = group["lr"] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                else:  # Rectified update
                    # calculate rho_t
                    state["rho_t"] = state["rho_inf"] - 2 * state[
                        "step"
                    ] * beta2 ** state["step"] / (1.0 - beta2 ** state["step"])

                    if (
                        state["rho_t"] > 4
                    ):  # perform Adam style update if variance is small
                        rho_inf, rho_t = state["rho_inf"], state["rho_t"]
                        rt = (
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * rho_inf
                            / (rho_inf - 4.0)
                            / (rho_inf - 2.0)
                            / rho_t
                        )
                        rt = math.sqrt(rt)

                        step_size = rt * group["lr"] / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:  # perform SGD style update
                        p.data.add_(-group["lr"], exp_avg)

        return loss


class AdamNorm(Optimizer):
    r"""Implements AdamNorm algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs
               WACV 2023
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        weight_decouple=False,
        fixed_decay=False,
        rectify=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(AdamNorm, self).__init__(params, defaults)

        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print("Weight decoupling enabled in AdamNorm")
            if self.fixed_decay:
                print("Weight decay fixed")
        if self.rectify:
            print("Rectification enabled in AdamNorm")
        if amsgrad:
            print("AMS enabled in AdamNorm")

    def __setstate__(self, state):
        super(AdamNorm, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                amsgrad = group["amsgrad"]

                # State initialization
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = (
                    torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if version_higher
                    else torch.zeros_like(p.data)
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_var"] = (
                    torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if version_higher
                    else torch.zeros_like(p.data)
                )
                state["exp_grad_norm"] = 0

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_var"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in reversed(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamNorm does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                beta1, beta2 = group["betas"]

                # State initialization
                if len(state) == 0:
                    state["rho_inf"] = 2.0 / (1.0 - beta2) - 1.0
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_var"] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )
                    state["exp_grad_norm"] = 0
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_var"] = (
                            torch.zeros_like(
                                p.data, memory_format=torch.preserve_format
                            )
                            if version_higher
                            else torch.zeros_like(p.data)
                        )

                # get current state variable
                exp_avg, exp_avg_var, exp_grad_norm = (
                    state["exp_avg"],
                    state["exp_avg_var"],
                    state["exp_grad_norm"],
                )

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                    else:
                        p.data.mul_(1.0 - group["weight_decay"])
                else:
                    if group["weight_decay"] != 0:
                        # grad.add_(group['weight_decay'], p.data)
                        grad.add_(p.data, alpha=group["weight_decay"])

                # Gradient Norm Correction
                grad_norm = LA.norm(grad)
                exp_grad_norm = 0.95 * exp_grad_norm + 0.05 * grad_norm
                if exp_grad_norm > grad_norm:
                    grad1 = grad * exp_grad_norm / grad_norm
                else:
                    grad1 = grad
                state["exp_grad_norm"] = exp_grad_norm.clone()

                # Update first and second moment running average
                # exp_avg.mul_(beta1).add_(1 - beta1, grad1)
                exp_avg.mul_(beta1).add_(grad1, alpha=1 - beta1)

                # exp_avg_var.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_var.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state["max_exp_avg_var"]
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (
                        max_exp_avg_var.add_(group["eps"]).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group["eps"])
                else:
                    denom = (
                        exp_avg_var.add_(group["eps"]).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group["eps"])

                if not self.rectify:
                    # Default update
                    step_size = group["lr"] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update
                    # calculate rho_t
                    state["rho_t"] = state["rho_inf"] - 2 * state[
                        "step"
                    ] * beta2 ** state["step"] / (1.0 - beta2 ** state["step"])

                    if (
                        state["rho_t"] > 4
                    ):  # perform Adam style update if variance is small
                        rho_inf, rho_t = state["rho_inf"], state["rho_t"]
                        rt = (
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * rho_inf
                            / (rho_inf - 4.0)
                            / (rho_inf - 2.0)
                            / rho_t
                        )
                        rt = math.sqrt(rt)

                        step_size = rt * group["lr"] / bias_correction1

                        p.data.addcdiv_(-step_size, exp_avg, denom)

                    else:  # perform SGD style update
                        p.data.add_(-group["lr"], exp_avg)

        return loss


class diffGrad(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(diffGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(diffGrad, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "diffGrad does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Previous gradient
                    state["previous_grad"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, previous_grad = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["previous_grad"],
                )
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # computer diffgrad coefficient
                diff = abs(previous_grad - grad)
                # print(diff)
                dfc = 1.0 / (1.0 + torch.exp(-diff))
                state["previous_grad"] = grad.clone()

                exp_avg1 = exp_avg * dfc

                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg1, denom)

        return loss


class diffGradNorm(Optimizer):
    r"""Implements diffGradNorm algorithm. Modified from diffGrad in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    reference: AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs
               WACV 2023
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(diffGradNorm, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(diffGradNorm, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "diffGradNorm does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    # Previous gradient
                    state["previous_grad"] = torch.zeros_like(p.data)
                    state["exp_grad_norm"] = 0

                exp_avg, exp_avg_sq, previous_grad, exp_grad_norm = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["previous_grad"],
                    state["exp_grad_norm"],
                )
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], p.data)

                # Gradient Norm Correction
                grad_norm = LA.norm(grad)
                exp_grad_norm = 0.95 * exp_grad_norm + 0.05 * grad_norm
                if exp_grad_norm > grad_norm:
                    grad1 = grad * exp_grad_norm / grad_norm
                else:
                    grad1 = grad
                state["exp_grad_norm"] = exp_grad_norm.clone()

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad1)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # computer diffgrad coefficient
                diff = abs(previous_grad - grad)
                # print(diff)
                dfc = 1.0 / (1.0 + torch.exp(-diff))
                state["previous_grad"] = grad.clone()

                exp_avg1 = exp_avg * dfc

                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg1, denom)

        return loss


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class Radam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        version=1,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd

        self.version = version

        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(Radam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Radam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("Radam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state["step"] += 1

                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class RadamNorm(Optimizer):
    r"""Implements RadamNorm algorithm. Modified from Radam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
    reference: AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs
               WACV 2023
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        version=1,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd

        self.version = version

        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RadamNorm, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RadamNorm, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RadamNorm does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    state["exp_grad_norm"] = 0

                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)
                    state["exp_grad_norm"] = 0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_grad_norm = state["exp_grad_norm"]
                beta1, beta2 = group["betas"]

                # Gradient Norm Correction
                grad_norm = LA.norm(grad)
                exp_grad_norm = 0.95 * exp_grad_norm + 0.05 * grad_norm
                if exp_grad_norm > grad_norm:
                    grad1 = grad * exp_grad_norm / grad_norm
                else:
                    grad1 = grad
                state["exp_grad_norm"] = exp_grad_norm.clone()

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad1)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state["step"] += 1

                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )

                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                    p.data.copy_(p_data_fp32)

                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )

                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss