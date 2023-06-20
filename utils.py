import os
import json

import torch
import torchvision

class Logger(object):
    def __init__(self, fp):
        self.fp = fp

    def log_info(self, content):
        print(content)
        print(content, file=self.fp)

def uniform_ab(size, a, b):
    # (a, b]
    return (a - b) * torch.rand(size) + b

def normal_ab(size, a, b):
    center = (b - a) / 2 + a
    return torch.clamp(torch.randn(size), -1, 0.9) * (b - center) + center


class Bernoulli(torch.autograd.Function):
    """
    Custom Bernouli function that supports gradients.
    The original Pytorch implementation of Bernouli function,
    does not support gradients.

    First-Order gradient of bernouli function with prbabilty p, is p.

    Inputs: Tensor of arbitrary shapes with bounded values in [0,1] interval
    Outputs: Randomly generated Tensor of only {0,1}, given Inputs as distributions.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):      
        pvals = ctx.saved_tensors
        return pvals[0] * grad_output

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GradSaver:
    def __init__(self):
        self.grad = -1
    
    def save_grad(self, grad):
        self.grad = grad


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_params(json_file):
    with open(json_file) as f:
        return json.load(f)

def get_batch(data_loader):
    while True:
        for batch in data_loader:
            yield batch

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
        #return super().__call__(tensor)

def my_scale(v, v_max, v_min, low=0, up=1):
    return (up - low) * (v - v_min) / max(1e-7, v_max - v_min) + low

def my_scale_inv(v, v_max, v_min, low=0, up=1):
    return (v - low) / (up - low) * max(1e-7, v_max - v_min) + v_min


class Record(object):
    def __init__(self):
        self.loss = 0
        self.count = 0

    def add(self, value):
        self.loss += value
        self.count += 1

    def mean(self):
        return self.loss / self.count


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.init_param()
        #self.init_dataset()

    def init_param(self):
        self.gpus = torch.cuda.device_count()

    def get_loader(self, dataset, shuffle=True, drop_last=True):
        data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.args.batch_size * self.gpus,
                            num_workers=int(self.args.num_workers),
                            shuffle=shuffle,
                            drop_last=drop_last
                        )
        return data_loader
