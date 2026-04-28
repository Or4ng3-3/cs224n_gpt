from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]


                ### TODO: Complete the implementation of AdamW here, reading and saving
                ###       your state in the `state` dictionary above.
                ###       The hyperparameters can be read from the `group` dictionary
                ###       (they are lr, betas, eps, weight_decay, as saved in the constructor).
                ###
                ###       To complete this implementation:
                ###       1. Update the first and second moments of the gradients.
                ###       2. Apply bias correction
                ###          (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                ###          also given in the pseudo-code in the project description).
                ###       3. Update parameters (p.data).
                ###       4. Apply weight decay after the main gradient-based updates.
                ###
                ###       Refer to the default project handout for more details.
                ### YOUR CODE HERE
                if(len(state) == 0):
                    state["step"] = 0
                    state["pre_m_t"] = torch.zeros_like(p.data)
                    state["pre_v_t"] = torch.zeros_like(p.data)

                beta_1 = group["betas"][0]
                beta_2 = group["betas"][1]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                state["step"] += 1
                m_t = state["pre_m_t"]
                v_t = state["pre_v_t"]

                # m_t = beta_1 * state["pre_m_t"] + (1-beta_1) * p.grad
                m_t.mul_(beta_1).add_(grad, alpha = 1 - beta_1)

                #v_t = beta_2 * state["pre_v_t"] + (1-beta_2) * p.grad * p.grad
                v_t.mul_(beta_2).addcmul_(grad, grad, value = 1 - beta_2) 
                
                # if we use add_ and mul_ which do not create new tensors in memory, we don't need to writeback.
                # state["pre_m_t"] = m_t
                # state["pre_v_t"] = v_t

                if(group["correct_bias"]):
                    alpha_t = alpha * ((1 - (beta_2 ** state["step"])) ** 0.5 / (1 - (beta_1 ** state["step"])))
                else:
                    alpha_t = alpha


                # theta_{t+1} = theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v_t}} + \epsilon} - \alpha \lambda \theta_t
                
                denom = (v_t.sqrt()).add_(eps)
                p.data.addcdiv_(m_t, denom, value = -alpha_t)

                if(weight_decay > 0):
                    p.data.add_ (p.data, alpha = -alpha * weight_decay)


                # p = p - alpha_t * state["weight_decay"] * (m_t / (v_t ** 0.5 + eps))


        return loss
