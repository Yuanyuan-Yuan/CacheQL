import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import utils
import models


class PMI(object):
    def __init__(self, args, logger):
        self.args = args
        self.fp = f
        self.logger = logger
        self.epoch = 0
        self.mse = nn.MSELoss().cuda()
        self.l1 = nn.L1Loss().cuda()
        self.bce = nn.BCELoss().cuda()
        self.bce_log = nn.BCEWithLogitsLoss().cuda()
        self.ce = nn.CrossEntropyLoss().cuda()
        self.init_model_optimizer()

    def init_model_optimizer(self):
        print('Initializing Model and Optimizer...')
        self.compressor = models.__dict__['key_decoder_%d' % self.args.key_length](
                nc=self.args.nc,
                size=self.args.size,
                dim=self.args.dim
                )

        self.critic_f = models.Estimator(
                nc=self.args.nc + 1,
                size=self.args.size,
                dim=self.args.key_length,
                hidden_dim=self.args.hidden_dim,
                output_dim=self.args.output_dim,
                n_layer=self.args.n_layer,
                mode='mlp'
                )

        self.logger.log_info('Compressor')
        self.logger.log_info(self.compressor)
        self.logger.log_info('Critic')
        self.logger.log_info(self.critic_f)

        self.compressor = self.compressor.cuda()
        self.critic_f = self.critic_f.cuda()

        self.optim = torch.optim.Adam(
                        list(self.critic_f.parameters()) +\
                        list(self.compressor.parameters()),
                        lr=self.args.lr,
                        betas=(self.args.beta1, 0.999)
                        )

    def save_model(self, path):
        print('Saving Model on %s ...' % (path))
        state = {
            'critic': self.critic_f.state_dict(),
            'compressor': self.compressor.state_dict(),
            'optim': self.optim.state_dict()
        }
        torch.save(state, path)

    def load_model(self, path):
        print('Loading Model from %s ...' % (path))
        ckpt = torch.load(path)
        self.critic_f.load_state_dict(ckpt['critic'])
        self.compressor.load_state_dict(ckpt['compressor'])
        self.optim.load_state_dict(ckpt['optim'])

    

    def zero_grad(self):
        self.compressor.zero_grad()
        self.critic_f.zero_grad()

    def set_train(self):
        self.compressor.train()
        self.critic_f.train()

    def set_eval(self):
        self.compressor.eval()
        self.critic_f.eval()

    def quantify(self, trace, secret):
        self.compressor.eval()
        self.critic_f.eval()
        with torch.no_grad():
            compressed = self.compressor(trace)
            # compressed = utils.Bernoulli.apply(compressed)            
            compressed = (compressed > 0.5).type_as(recovered)
            pos_scores = self.critic_f((recovered, secret))
            pos_logits = F.sigmoid(pos_scores)
            pmi = torch.log(pos_logits / (1 - pos_logits)) / np.log(2)
            pmi /= (np.log(self.m_size) / np.log(2))
            pmi = torch.clamp(pmi, 0, 1).mean() * self.args.key_length
            return pmi

    def process_grad(self, trace, secret):
        self.zero_grad()
        trace = trace.to(config.DEVICE)
        secret = secret.to(config.DEVICE)
        batch_size = trace.size(0)
        grad_saver = utils.GradSaver()
        compressed = self.compressor(trace)
        compressed = utils.Bernoulli.apply(compressed)
        pos_scores = self.critic_f((compressed, secret))
        ones = torch.ones(pos_scores.size()).to(config.DEVICE)
        pos_logits = F.sigmoid(pos_scores)
        loss = self.bce_log(pos_scores, ones)
        loss.backward()
        gradient = grad_saver.grad.detach().abs().view(batch_size, -1)
        return gradient

    def fit(self, data_loader):
        with torch.autograd.set_detect_anomaly(True):
            self.epoch += 1
            self.set_train()
            record = utils.Record()
            start_time = time.time()
            for i, (trace, secret, name) in enumerate(tqdm(data_loader)):
                self.zero_grad()

                trace = trace.cuda()
                secret = secret.cuda()
                batch_size = trace.size(0)

                random_index = torch.randperm(batch_size).long()
                compressed = self.compressor(trace)
                compressed = utils.Bernoulli.apply(compressed)
                pos_scores = self.critic_f((compressed, secret))
                neg_scores = self.critic_f((compressed, secret[random_index]))
                ones = torch.ones(pos_scores.size()).cuda()
                zeros = torch.zeros(neg_scores.size()).cuda()
                pos_logits = F.sigmoid(pos_scores) # pos_logit \in [0.5, bs / (bs + 1)]
                neg_logits = F.sigmoid(neg_scores) # neg_logit \in [1 / (bs + 1), 0.5]

                loss = self.bce_log(pos_scores, ones) + self.bce_log(neg_scores, zeros)

                with torch.no_grad():
                    pmi = torch.log(pos_logits / (1 - pos_logits)) / np.log(2)
                    pmi /= (np.log(batch_size) / np.log(2))
                    pmi = torch.clamp(pmi, 0, 1).mean() * self.args.key_length
                record.add(pmi.item())

                loss.backward()
                self.optim.step()
            self.logger.log_info('----------------------------------------')
            self.logger.log_info('Fitting Epoch: %d' % self.epoch)
            self.logger.log_info('Costs Time: %.2f s' % (time.time() - start_time))
            self.logger.log_info('MI: %f' % (record.mean()))
            self.logger.log_info('----------------------------------------')

    def validate(self, data_loader):
        with torch.no_grad():
            self.set_eval()
            record = utils.Record()
            start_time = time.time()
            for i, (trace, secret, name) in enumerate(tqdm(data_loader)):
                trace = trace.cuda()
                secret = secret.cuda()
                batch_size = trace.size(0)
                m_size = self.args.batch_size

                random_index = torch.randperm(batch_size).long()
                compressed = self.compressor(trace)
                # compressed = utils.Bernoulli.apply(compressed)
                compressed = (compressed > 0.5).type_as(compressed)
                pos_scores = self.critic_f((compressed, secret))
                neg_scores = self.critic_f((compressed, secret[random_index]))
                ones = torch.ones(pos_scores.size()).cuda()
                zeros = torch.zeros(neg_scores.size()).cuda()
                pos_logits = F.sigmoid(pos_scores)
                neg_logits = F.sigmoid(neg_scores)

                loss = self.bce_log(pos_scores, ones) + self.bce_log(neg_scores, zeros)

                pmi = torch.log(pos_logits / (1 - pos_logits)) / np.log(2)
                pmi /= (np.log(m_size) / np.log(2))
                pmi = torch.clamp(pmi, 0, 1).mean() * self.args.key_length
                record.add(pmi.item())
            self.logger.log_info('----------------------------------------')
            self.logger.log_info('Validation.')
            self.logger.log_info('Costs Time: %.2f s' % (time.time() - start_time))
            self.logger.log_info('MI: %f' % (record.mean()))
            self.logger.log_info('----------------------------------------')



if __name__ == '__main__':
    import sys
    import argparse
    import random
    import json

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    
    from dataset import *
    from params import Params

    args = Params().parse()

    if len(args.side) > 0:
        args.exp_name = 'mi-%s-%s-%s' % (args.software, args.setting, args.side)
    else:
        args.exp_name = 'mi-%s-%s' % (args.software, args.setting)
    print(args.exp_name)

    args.npz_dir = os.path.join(config.trace_dir, args.setting, args.software)
    
    if 'rsa' in args.software:
        args.key_dir = config.rsa_key_npz_dir
    elif 'aes' in args.software:
        args.key_dir = config.aes_key_npz_dir
    
    (args.size, args.nc) = args.PADLENGTH['%s-%s' % (args.software, args.setting)]
    
    fp = open(os.path.join(config.output_dir, args.exp_name, 'log.txt'), 'a')
    logger = utils.Logger(fp)

    manual_seed = random.randint(1, 10000)
    logger.log_info('Manual Seed: %d' % manual_seed)

    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    utils.make_path(os.path.join(config.output_dir, args.exp_name))

    args.ckpt_dir = os.path.join(config.output_dir, args.exp_name, 'ckpt')
    utils.make_path(args.ckpt_dir)

    with open(os.path.join(config.output_dir, args.exp_name, 'args.json'), 'a') as f:
        json.dump(args.__dict__, f)

    loader = utils.DataLoader(args)

    fit_dataset = RSADatasetMulti(
                    args,
                    key_dir=args.key_dir, 
                    npz_dir=args.npz_dir, 
                    key_split='fit',
                    npz_split_list=['1_fit', '2_fit', '3_fit', '4_fit']
                )

    val_dataset = RSADatasetMulti(
                    args,
                    key_dir=args.key_dir, 
                    npz_dir=args.npz_dir, 
                    key_split='val',
                    npz_split_list=['1_val']
                )

    fit_loader = loader.get_loader(fit_dataset)
    val_loader = loader.get_loader(val_dataset, False)

    engine = PMI(args, logger)

    for i in range(engine.epoch, args.num_epoch):
        engine.fit(fit_loader)
        if i % args.test_freq == 0:
            engine.validate(val_loader)
            engine.save_model((args.ckpt_dir + '%03d.pth') % (i + 1))
    engine.save_model((args.ckpt_dir + 'final.pth'))

