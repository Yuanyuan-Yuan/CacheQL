import os
import time
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

import utils
import config

BASE = 0

def integrated_gradients(compressor, critic_f, trace, secret, steps=50, baseline=None):
    if baseline is None:
        baseline = BASE * torch.ones(trace.size()).cuda()
    bce_log = nn.BCEWithLogitsLoss().cuda()
    accu_traces = [baseline + (float(i) / steps) * (trace - baseline) for i in range(0, steps + 1)]
    accu_traces = torch.cat(accu_traces, 0).cuda()
    accu_traces.requires_grad = True
    repeat_key = torch.cat([secret for _ in range(accu_traces.size(0))]).cuda()
    
    compressor.zero_grad()
    critic_f.zero_grad()
    compressed = compressor(accu_traces)
    compressed = utils.Bern.apply(compressed)
    pos_score = critic_f((compressed, repeat_key))
    ones = torch.ones(pos_score.size()).cuda()
    loss = bce_log(pos_score, ones)
    loss.backward()
    grads = accu_traces.grad
    avg_grads = torch.mean(grads, 0)
    avg_grads = avg_grads.unsqueeze(0)
    delta_X = (trace - baseline)
    integrated_grad = delta_X * avg_grads
    return integrated_grad

def set_mask(mask, v, index_list):
    assert v in [0, 1]
    mask_shape = mask.shape
    mask = mask.view(-1)
    mask[index_list] = v
    mask = mask.view(mask_shape)
    return mask

def Monte_Carlo_round(game, trace, key, index_list, existing_shapley, count, device=torch.device('cuda')):
    n_player = len(index_list)
    player2index = {}
    for i, index in enumerate(index_list):
        player2index[index] = i
    permutation = np.array(range(n_player), 'int32')
    permutation = np.random.permutation(permutation).tolist()
    mask = torch.ones(trace.shape[1:]).to(device)
    mask = set_mask(mask, 0, index_list)
    if existing_shapley is None or count == 0:
        existing_shapley = np.array([0] * n_player, 'float64')
    updated_shapley = np.array([0] * n_player, 'float64')
    prev_value = game(mask * trace + (1 - mask) * BASE, key)
    # Here, the game takes (trace, secret) as input and quantifies the leakage
    for i in permutation:
        player_index = index_list[i]
        mask = set_mask(mask, 1, player_index)
        cur_value = game(mask * trace + (1 - mask) * BASE, key)
        gain_value = cur_value - prev_value
        prev_value = cur_value
        updated_shapley[player2index[player_index]] = gain_value.item()
    existing_shapley = (count * existing_shapley + updated_shapley) / (count + 1)
    return existing_shapley, count + 1


class Shapley(PMI):
    def localize(self, data_loader, inst_dir, save_dir):
        self.set_eval()
        start_time = time.time()
        count_dict = {}
        leakage_dict = {}
        trace_dict = {}

        for _, (trace, secret, name_list) in enumerate(tqdm(data_loader)):
            if args.use_IG:
                IG = integrated_gradients(self.compressor, self.critic_f, trace, secret)
                gradient = IG.view(batch_size, -1).abs().detach().cpu()
            else:
                gradient = self.process_grad(trace, secret)

            assert len(gradient) == len(name_list)

            det_MI = self.quantify(trace, secret)[0].item()
            if det_MI <= 0:
                continue
            
            grad_index = torch.argsort(gradient, dim=-1, descending=True)
            with torch.no_grad():
                candidate_list = grad_index[0].cpu().numpy().tolist()
                cur_MI = det_MI
                idx_cnt = 0
                idx_list = []
                flat_trace = copy.deepcopy(trace[0]).view(1, -1)
                while cur_MI > 0 and idx_cnt < args.key_length // 2:
                    flat_trace[0][candidate_list[idx_cnt]] = BASE
                    cur_trace = flat_trace.view(1, args.nc, args.size, args.size)
                    cur_MI = self.quantify(cur_trace, secret)[0].item()
                    idx_list.append(candidate_list[idx_cnt])
                    idx_cnt += 1
                if cur_MI >= det_MI:
                    continue
                valid_cnt += 1
                
                (existing_shapley, count) = (None, 0)
                existing_shapley, count = MC_round(self.quantify, trace, secret, idx_list, existing_shapley, count, self.args.key_length)
                record_list = [existing_shapley]
                for _ in range(100):
                    existing_shapley, count = Monte_Carlo_round(self.quantify, trace, secret, idx_list, existing_shapley, count, self.args.key_length)
                    diff = np.abs(record_list[-1] - existing_shapley)
                    if diff.max() < 0.5:
                        break
                    record_list.append(existing_shapley)

                name = name_list[0]
                with open(inst_dir + name + '.out', 'r') as f:
                    inst = f.readlines()
                trace_dict[name] = {}
                selected_addr = []
                for j, idx in enumerate(idx_list):
                    if idx >= len(inst)-1:
                        continue
                    selected_inst = inst[idx]
                    [func_name, assembly, content] = selected_inst.strip().split('; ')
                    [ins_addr, op, mem_addr] = content.strip().split(' ')
                    ins_addr = str(hex(int(ins_addr.replace(':', ''))))
                    res_key = '%s; %s; %s' % (func_name, assembly, ins_addr)
                    selected_addr.append(res_key)
                    if res_key not in leakage_dict.keys():
                        leakage_dict[res_key] = existing_shapley[j]
                    else:
                        leakage_dict[res_key] += existing_shapley[j]
                    trace_dict[name][res_key] = existing_shapley[j]
                for addr in selected_addr:
                    if addr in count_dict.keys():
                        count_dict[addr] += 1
                    else:
                        count_dict[addr] = 1
        for res_key in leakage_dict.keys():
            leakage_dict[res_key] /= count_dict[res_key]

        leakage_save = {k: leakage_dict[k] for k in sorted(leakage_dict, key=leakage_dict.get, reverse=True)}
        count_save = {k: count_dict[k] for k in sorted(count_dict, key=count_dict.get, reverse=True)}
        with open(save_dir + 'active_player.json', 'w') as f:
            json.dump(leakage_save, f, indent=2)
        with open(save_dir + 'trace_localize.json', 'w') as f:
            json.dump(trace_dict, f, indent=2)
        with open(save_dir + 'count.json', 'w') as f:
            json.dump(count_save, f, indent=2)

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

    args.batch_size = 1

    args.ckpt_dir = os.path.join(config.output_dir, args.exp_name, 'ckpt')
    args.loc_dir = os.path.join(config.output_dir, args.exp_name, 'loc')
    utils.make_path(args.loc_dir)
    # Note: remember to specify `exp_name`

    loader = utils.DataLoader(args)

    val_dataset = RSADatasetMulti(
                    args,
                    key_dir=args.key_dir, 
                    npz_dir=args.npz_dir, 
                    key_split='val',
                    npz_split_list=['1_val']
                )

    val_loader = loader.get_loader(val_dataset, False)

    engine = Shapley(args)
    engine.m_size = 128
    # Note: when doing localization, use `m_size` value as the `batch_size` when training the model

    model_path = os.path.join(args.ckpt_dir, 'final.pth')
    engine.load_model(model_path)

    split = '1_val'
    inst_dir = os.path.join(config.ins_addr, args.setting, args.software, split)
    result_dir = os.path.join(args.loc_dir, split)

    engine.localize(val_loader, ins_addr, result_dir)


