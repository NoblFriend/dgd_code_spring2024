import torch

class EF21:
    def __init__(self, params):
        self._params = params
        self._comp_err = {name: torch.zeros_like(param) for name, param in params}
        self.compression_operator = lambda x: x

    def step(self):
        for name, param in self._params:
            self._comp_err[name] += self.compression_operator(param.grad.detach() - self._comp_err[name])
            param.grad = self._comp_err[name]

    def get_compression_errors(self):
        return {name: err.clone() for name, err in self._comp_err.items()}

    def load_compression_errors(self, comp_err_dict):
        for name, err in comp_err_dict.items():
            if name in self._comp_err:
                self._comp_err[name].copy_(err)