import torch
import torch.nn as nn

class GD(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(GD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                with torch.no_grad():
                    grad = p.grad
                    if weight_decay != 0:
                        grad.add_(p, alpha=weight_decay)
                    p.sub_(grad, alpha=lr)
