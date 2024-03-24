import torch

class Worker:
    def __init__(self, model, X, y, criterion, compress_op=lambda x: x):
        self.model = model 
        self.X = X
        self.y = y
        self.criterion = criterion
        self.compress_op = compress_op
        self.saved_gradient = {name: torch.zeros_like(param.data) for name, param in model.named_parameters()}

    def get_gradient_ef21(self):
        self.model.zero_grad()
        
        output = self.model(self.X)
        loss = self.criterion(output, self.y)
        loss.backward()
        
        corrected_gradients = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                corrected_grad = self.compress_op(param.grad - self.saved_gradient[name])
                self.saved_gradient[name] += corrected_grad
                corrected_gradients[name] = corrected_grad
        
        return corrected_gradients
