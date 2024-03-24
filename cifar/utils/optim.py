import torch
import torch.nn as nn

class GD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        super(GD, self).__init__(params, defaults={'lr': lr})

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data = p.data - group['lr'] * p.grad.data
