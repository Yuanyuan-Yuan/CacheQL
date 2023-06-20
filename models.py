import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

def conv(nc, size, dim):
    """Create a conv net for dimension reduction."""
    nf = 64
    n_layer = int(np.log(size) / np.log(2)) - 1
    conv_list = [dcgan_conv(nc, nf)]
    if n_layer > 2:
        for i in range(n_layer - 2):
            conv_list += [dcgan_conv(min(2**i, 8) * nf, min(2**(i + 1), 8) * nf)]
    conv_list += [nn.Sequential(
                    nn.Conv2d(min(2**(n_layer - 2), 8) * nf, dim, 4, 1, 0),
                    nn.BatchNorm2d(dim),
                    # nn.Tanh()
                )]
    return conv_list

def dconv(nc, size, dim):
    nf = 64
    n_layer = int(np.log(size) / np.log(2)) - 1
    dconv_list = [nn.Sequential(
                nn.ConvTranspose2d(dim, nf * min(2**(n_layer - 2), 8), 4, 1, 0),
                nn.BatchNorm2d(nf * min(2**(n_layer - 2), 8)),
                nn.LeakyReLU(0.2)
                )]
    if n_layer > 2:
        for i in range(n_layer - 2 - 1, -1, -1):
            dconv_list += [dcgan_upconv(nf * min(2**(i + 1), 8), nf * min(2**i, 8))]
    dconv_list += [nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                # nn.Tanh() # --> [-1, 1]
                # nn.Sigmoid() # --> [0, 1]
                )]
    return dconv_list

# def fc(dim, hidden_dim, output_dim, n_layer):
#     """Create a sequence of fc net."""
#     fc_list = [nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2)]
#     for _ in range(n_layer):
#         fc_list += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)]
#     fc_list += [nn.Linear(hidden_dim, output_dim)]
#     return fc_list

def fc(dim, hidden_dim, output_dim, n_layer):
    """Create a sequence of fc net."""
    fc_list = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layer):
        fc_list += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    fc_list += [nn.Linear(hidden_dim, output_dim)]
    return fc_list


class net(nn.Module):
    def __init__(self, nc, size, dim, hidden_dim, output_dim, n_layer):
        super(net, self).__init__()
        self.dim = dim
        self.conv = nn.Sequential(*conv(nc, size, dim))
        self.fc = nn.Sequential(*fc(dim, hidden_dim, output_dim, n_layer))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.dim)
        out = self.fc(out)
        return out

class mlp(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, n_layer):
        super(mlp, self).__init__()
        self.fc = nn.Sequential(*fc(dim, hidden_dim, output_dim, n_layer))

    def forward(self, x):
        out = self.fc(x)
        return out

class net2net(nn.Module):
    def __init__(self, nc, size, dim):
        super(net2net, self).__init__()
        self.dim = dim
        self.conv = nn.Sequential(*conv(nc, size, dim))
        self.dconv = nn.Sequential(*dconv(nc, size, dim))
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv(x)
        out = self.tanh(out)
        out = out.view(-1, self.dim, 1, 1)
        out = self.dconv(out)
        out = self.tanh(out)
        return out


class Estimator(nn.Module):
    def __init__(self, nc, size, dim, hidden_dim, output_dim, n_layer, mode='mlp'):
        super(Estimator, self).__init__()
        # output is scalar score
        if mode == 'conv':
            self._f = net(nc * 2, size, dim, hidden_dim, output_dim, n_layer)
        elif mode == 'mlp':
            self._f = mlp(dim * 2, hidden_dim, output_dim, n_layer)
        else:
            raise NotImplementedError

    def forward(self, inputs, is_cat=False):
        if is_cat:
            scores = self._f(inputs)
        else:
            (x, y) = inputs
            xy_pairs = torch.cat((x, y), dim=1)
            # xy is [batch_size, x_dim + y_dim]
            scores = self._f(xy_pairs)
        return scores

    def forward_grad(self, x, y, grad_saver):
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        xy_pairs = torch.cat((x, y), dim=1)
        # xy is [batch_size, x_dim + y_dim]
        scores = self._f(xy_pairs)
        return scores


class key_decoder_1024(nn.Module):
    def __init__(self, nc, size, dim):
        super(key_decoder_1024, self).__init__()
        self.dim = dim
        nf = 64
        self.conv = nn.Sequential(
                *conv(nc=nc, size=size, dim=dim)
                )
        self.fc = nn.Sequential(
                nn.Linear(self.dim, 1024),
                nn.Sigmoid()
                )

    def forward(self, x):
        out = self.conv(x)
        if isinstance(out, tuple):
            batch_size = out[0].size(0)
            out = (out[0].view(batch_size, -1), out[1].view(batch_size, -1))
        else:
            batch_size = out.size(0)
            out = out.view(batch_size, -1)
        key = self.fc(out)
        return key

    def forward_grad(self, x, grad_saver):
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        batch_size = x.size(0)
        out = self.conv(x)
        key = self.fc(out.view(batch_size, -1))
        return key

class key_decoder_128(nn.Module):
    def __init__(self, nc, size, dim):
        super(key_decoder_128, self).__init__()
        self.dim = dim
        nf = 64
        self.conv = nn.Sequential(
                *conv(nc=nc, size=size, dim=dim)
                )
        self.fc = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.Sigmoid()
                )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv(x)
        key = self.fc(out.view(batch_size, -1))
        return key

    def forward_grad(self, x, grad_saver):
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        batch_size = x.size(0)
        out = self.conv(x)
        key = self.fc(out.view(batch_size, -1))
        return key

class image_decoder_128(nn.Module):
    def __init__(self, nc, size, dim, image_nc=3, image_size=128):
        super(image_decoder_128, self).__init__()
        self.dim = dim
        nf = 64
        self.conv = nn.Sequential(
                *conv(nc=nc, size=size, dim=dim)
                )
        self.dec = nn.Sequential(
                *dconv(nc=image_nc, size=image_size, dim=dim),
                nn.Tanh()
                )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv(x)
        image = self.dec(out.view(batch_size, -1, 1, 1))
        return image

    def forward_grad(self, x, grad_saver):
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        batch_size = x.size(0)
        out = self.conv(x)
        image = self.dec(out.view(batch_size, -1, 1, 1))
        return image

class key_decoder_128_fc(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(key_decoder_128_fc, self).__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
                )
        self.fc2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
                )
        self.fc3 = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid()
                )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        return x3

    def forward_grad(self, x, grad_saver):
        x.requires_grad = True
        x.register_hook(grad_saver.save_grad)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x2)
        x3 = self.fc3(x2)
        return x3