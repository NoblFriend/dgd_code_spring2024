import torch

class EF21:
    def __init__(self, params):
        self._params = params
        self._comp_err = {name: torch.zeros_like(param) for name, param in params}
        self.compression_operator = lambda x: x

    def step(self):
        for name, param in self._params:
            comp_grad = self.compression_operator(param.grad.detach() - self._comp_err[name])
            self._comp_err[name] += comp_grad
            param.grad = comp_grad