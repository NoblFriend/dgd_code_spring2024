import torch

class EF21:
    def __init__(self, params):
        self._params = params
        self._comp_err = {name: torch.zeros_like(param) for name, param in params}
        self.comp_operator = lambda x: x
        self.comp_factors = []

    def step(self):
        sent_size = []
        totat_size = []
        for name, param in self._params:
            self._comp_err[name] += self.comp_operator(param.grad.detach() - self._comp_err[name])
            sent_size.append(self.comp_operator.data_size)
            totat_size.append(param.grad.numel())
            param.grad = self._comp_err[name]
        self.comp_factors.append(sum(sent_size)/sum(totat_size))
        

    def get_compression_errors(self):
        return {name: err.clone() for name, err in self._comp_err.items()}

    def load_compression_errors(self, comp_err_dict):
        for name, err in comp_err_dict.items():
            if name in self._comp_err:
                self._comp_err[name].copy_(err)