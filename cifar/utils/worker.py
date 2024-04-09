from importlib.metadata import distribution
import torch
import copy


class Worker:
    def __init__(self, model, loader, criterion, compress_op=lambda x: x):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.compress_op = compress_op
        self.saved_gradient = {name: torch.zeros_like(
            param.data) for name, param in model.named_parameters()}

    def gradient_generator(self):
        for data, target in self.loader:
            # data, target = data.to('mps'), target.to('mps')

            self.model.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            corrected_gradients = {}

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    corrected_grad = self.compress_op(
                        param.grad - self.saved_gradient[name])
                    self.saved_gradient[name] += corrected_grad
                    corrected_gradients[name] = corrected_grad

            yield corrected_gradients


def setup_workers(original_model, criterion, dataset, num_workers, batch_size):
    dataset_size = len(dataset)
    sizes = [dataset_size // num_workers for _ in range(num_workers)]
    sizes[-1] += dataset_size - sum(sizes)  
    workers = []
    for subset in torch.utils.data.random_split(dataset, sizes):
        worker_model = copy.deepcopy(original_model)
        worker_model = original_model
        worker_loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=False, num_workers=0)
        worker = Worker(worker_model, worker_loader, criterion)
        workers.append(worker)
    return workers

def update_worker_models(workers, global_model):
    for worker in workers:
        worker.model = copy.deepcopy(global_model)
