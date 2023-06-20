import os
import numpy as np
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import utils
import mutual

__all__ = [
    'RSADatasetMulti',
    'RSADatasetMultiDec',
    'RSADatasetMultiPre',
    'RSADatasetMultiPP',
    'ImageDatasetMulti',
]

def scale(x):
    x -= x.min()
    x /= x.max()
    return x

def norm(x):
    x = scale(x)
    x -= 0.5
    x /= 0.5
    return x

def debias(x):
    x -= x.min()
    return x

class FullToSide(object):
    def __init__(self):
        pass

    def __call__(self, addr, side):
        if side in ['full', '']:
            return addr
        if side == 'cacheline':
            cacheline = self.to_cacheline20(np.abs(addr))
            # return (cacheline - 32) / 32
            return cacheline
        if side == 'cachebank':
            cachebank = self.to_cachebank(np.abs(addr))
            # return (cachebank - 512) / 512
            return cachebank
    def to_cacheline(self, addr, ASLR=False):
        return (addr & 0xF_FFFF) >> 6

    def to_cachebank(self, addr, ASLR=False):
        return (addr & 0xF_FFFF) >> 2

class RSADatasetMulti(Dataset):
    def __init__(self, args, key_dir, npz_dir, key_split, npz_split_list):
        super(RSADatasetMulti).__init__()
        self.args = args
        self.key_dir = os.path.join(key_dir, key_split)

        self.tool = FullToSide()

        self.npz_path_list = []
        for npz_split in npz_split_list:
            split_dir = os.path.join(npz_dir, npz_split)
            name_list = sorted(os.listdir(split_dir))
            for name in name_list:
                self.npz_path_list.append(
                    os.path.join(split_dir, name)
                    )
        
        print('Total %d %s data.' % (len(self.npz_path_list), key_split))

    def __len__(self):
        return len(self.npz_path_list)

    def load_trace(self, npz_path):
        npz_trace = np.load(npz_path)
        trace = npz_trace['arr_0']
        return trace

    def __getitem__(self, index):
        npz_path = self.npz_path_list[index]
        npz_name = npz_path.split(os.sep)[-1]
        key_name = npz_name

        trace = self.load_trace(npz_path)
        trace = self.tool(trace, self.args.side)
        trace = trace.astype(np.float32)
        trace = torch.from_numpy(trace)

        if self.args.use_norm:
            trace = norm(trace)
        elif self.args.use_bias:
            trace = debias(trace)
        trace = trace.view([self.args.nc, self.args.size, self.args.size])

        npz_key = np.load(os.path.join(self.key_dir, key_name))
        key = npz_key['arr_0']
        key = key.astype(np.float32)
        key = torch.from_numpy(key) # (1024, )
        return trace, key, key_name.split('.')[0]

class RSADatasetMultiDec(RSADatasetMulti):
    def load_trace(self, npz_path):
        npz_trace = np.load(npz_path)
        trace = npz_trace['arr_0'][self.args.start_index:]
        return trace

class RSADatasetMultiPre(RSADatasetMulti):
    def load_trace(self, npz_path):
        npz_trace = np.load(npz_path)
        trace = npz_trace['arr_0'][:self.args.end_index]
        return trace

class RSADatasetMultiPP(RSADatasetMulti):
    def load_trace(self, npz_path):
        npz_trace = np.load(npz_path)
        trace = npz_trace['arr_0']
        length = (len(trace) // 16) * self.args.repeat_num
        trace = trace[:length]
        return trace


class ImageDatasetMulti(Dataset):
    def __init__(self, args, image_dir, npz_dir, image_split, npz_split_list, data_num=None):
        super(ImageDatasetMulti).__init__()
        self.args = args
        self.image_dir = os.path.join(image_dir, split)
        self.image_dir = image_dir + ('%s/' % image_split)
        self.tool = FullToSide()

        self.transform = transforms.Compose([
                       transforms.Resize(args.image_size),
                       transforms.CenterCrop(args.image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               ])

        self.npz_path_list = []
        for npz_split in npz_split_list:
            split_dir = os.path.join(npz_dir, npz_split)
            name_list = sorted(os.listdir(split_dir))
            for name in name_list:
                self.npz_path_list.append(
                    os.path.join(split_dir, name)
                    )
        if data_num is not None:
            self.npz_path_list = self.npz_path_list[:data_num]
        
        print('Total %d %s data.' % (len(self.npz_path_list), image_split))

    def __len__(self):
        return len(self.npz_path_list)

    def __getitem__(self, index):
        npz_path = self.npz_path_list[index]
        npz_name = npz_path.split(os.sep)[-1]
        prefix = npz_name.split('.')[0]
        image_name = prefix + '.jpg'

        npz_trace = np.load(npz_path)
        trace = npz_trace['arr_0']
        trace = self.tool(trace, self.args.side)
        trace = trace.astype(np.float32)

        trace = torch.from_numpy(trace)
        if self.args.use_norm:
            trace = norm(trace)
        elif self.args.use_bias:
            trace = debias(trace)
        trace = trace.view([self.args.nc, self.args.size, self.args.size])
        
        image = Image.open(os.path.join(self.image_dir, image_name))
        image = self.transform(image)
        return trace, image, prefix

if __name__ == '__main__':
    pass