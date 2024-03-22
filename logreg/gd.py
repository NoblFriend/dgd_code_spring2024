from gc import enable
import numpy as np
import matplotlib.pyplot as plt


class Func:
    def __init__(self, func, grad):
        self.f_ = func
        self.grad = grad
        self.hess = None

    def __call__(self, x):
        return self.f_(x)

class Worker:
    def __init__(self, func):
        self.f : Func = func
        self.w = None
        self.compress_op = lambda x: x
        self.error = 0

    def get_gradient(self):
        return self.compress_op(self.f.grad(self.w))
    
    def get_gradient_ef21(self):
        c = self.compress_op(self.f.grad(self.w) - self.error)
        self.error += c
        return c


class DistributedGD:
    def __init__(self, workers, step):
        self.workers : list[Worker] = workers
        if isinstance(step, (int, float)):
            self.step = lambda *_: step
        else:
            self.step = step
        self.history = []

    def reset(self):
        self.history = []
        for worker in self.workers:
            worker.w = None
            worker.error = 0

    def run(self, num_iter, w0):
        for worker in self.workers:
            worker.w = w0
        w = w0
        for k in range(num_iter):
            mean_grad = np.mean([worker.get_gradient() for worker in self.workers], axis=0)
            w = w - self.step(w, k) * mean_grad

            for _, worker in enumerate(self.workers):
                worker.w = w

            self.history.append(w)

    def run_ef21(self, num_iter, w0):
        for worker in self.workers:
            worker.w = w0
        w = w0
        mean_grad = 0
        for k in range(num_iter):
            mean_add = np.mean([worker.get_gradient_ef21() for worker in self.workers], axis=0)
            mean_grad += mean_add
            w = w - self.step(w, k) * mean_grad

            for _, worker in enumerate(self.workers):
                    worker.w = w

            self.history.append(w)



def create_worker_func(f : Func, *args):
    def wrapped_f(w):
        return f(w, *args)

    def wrapped_grad(w):
        return f.grad(w, *args)

    return Func(wrapped_f, wrapped_grad)
