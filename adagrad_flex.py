# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
import pdb

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

from fairseq.optim import FairseqOptimizer, register_optimizer

logger = logging.getLogger(__name__)

def gmean(input_x):
    log_x = torch.log(input_x.flatten())
    return torch.exp(torch.mean(log_x))

class AdaGradFlex(torch.optim.Optimizer):
    """
    Adagrad with coordinate-wise flex statistics.
    """
    def __init__(
        self, params: _params_t, 
        lr: float = 1.0,
        momentum: float = 0, 
        log_every: int = 0,
        weight_decay: float = 0.0,
        eps: float = 1e-20,
        decouple: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if momentum < 0:
            raise ValueError(f"Momentum {momentum} must be non-negative")

        print(f"Weight decay: {weight_decay}")

        defaults = dict(lr=lr, 
            momentum=momentum,
            eps=eps, 
            weight_decay=weight_decay,
            log_every=log_every,
            k = 0,
            numerator_weighted=0.0,
            decouple=decouple)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        momentum = group['momentum']
        ck = 1 - momentum
        
        log_every = group['log_every']

        for group in self.param_groups:
            eps = group["eps"]
            k = group['k']
            decay = group['weight_decay']
            decouple = group['decouple']
            lr = group['lr']

            below_one = 0
            total = 0

            ######
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                
                if "alphak" not in state:
                    state["alphak"] = torch.zeros_like(p.data).detach()
                    #state["gsq"] = torch.zeros_like(p.data).detach()
                    state["gmax"] = torch.zeros_like(p.data).detach()
                    state['sk'] = torch.zeros_like(p.data).detach()
                    if momentum > 0:
                        state["z"] = torch.clone(p.data).detach()
                    state['flex'] = torch.zeros_like(p.data).detach()

                sk = state['sk']
                #gsq = state['gsq']
                alphak = state['alphak']
                gmax = state['gmax']
                flex = state['flex']

                if grad.is_sparse: 
                    grad = grad.to_dense()

                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                flex.add_(grad*grad).sub_(grad * sk)

                alphak.copy_(alphak.fmax(flex))
                gmax.copy_(gmax.fmax(grad.abs()))
                sk.add_(grad)

                if decay != 0 and decouple:
                    p_old = p.data.clone()

                if momentum > 0:
                    z = state['z']
                    z.sub_(grad.div(torch.sqrt(gmax*gmax + alphak) + eps), alpha=lr)
                    p.data.mul_(1-ck).add_(z, alpha=ck)

                    if decay != 0 and decouple:
                        z.add_(p_old, alpha=-decay * lr)
                else:
                    p.data.sub_(grad.div(torch.sqrt(gmax*gmax + alphak) + eps), alpha=lr)

                    if decay != 0 and decouple:
                        p.data.add_(p_old, alpha=-decay * lr)

            
                ### Logging
            #     below_one += ((alphak+eps)/(gmax*gmax + eps) < 1).sum().item()
            #     total += grad.numel()

            # if k % 50 == 0 and k > 0:
            #     print(f"fraction below 1: {below_one/total}")
            #     ratio = (alphak+eps)/(gmax*gmax + eps)
            #     print(f"mean: {ratio.mean()} gmean: {gmean(ratio)} std: {ratio.std()}")
            #     qs = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
            #     quantiles = torch.quantile(ratio, q=torch.tensor(qs).cuda())
            #     print(f"quantiles: {list(zip(qs, quantiles))}")

            group['k'] = k + 1
        return loss
