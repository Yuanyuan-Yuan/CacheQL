

if __name__ == '__main__':
    import os
    import sys
    import argparse
    import random
    import json

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import utils
    from data_loader import DataLoader
    
    from dataset import *
    from params import Params

    args = Params().parse()

    if len(args.side) > 0:
        args.exp_name = 'pmi-%s-%s-%s' % (args.software, args.setting, args.side)
    else:
        args.exp_name = 'pmi-%s-%s' % (args.software, args.setting)
    print(args.exp_name)

    args.npz_dir = os.path.join(config.trace_dir, args.setting, args.software)

    if 'rsa' in args.software:
        args.key_dir = config.rsa_key_npz_dir
    elif 'aes' in args.software:
        args.key_dir = config.aes_key_npz_dir
    
    PADLENGTH = {
        'aes_openssl_3.0.0': {
            # 1, 2, 4, 8, 16
            [(64, 2), (128, 1), (128, 2), (128, 4), (128, 8)]
        },
        'rsa_openssl_0.9.7c': {
            # 1, 2, 4, 8, 16
            [(64, 2), (128, 1), (128, 2), (128, 4), (128, 8)]
        },
    }

    (args.size, args.nc) = PADLENGTH['%s' % args.software][int(np.log(args.repeat_num)/np.log(2))]
    
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

    fit_dataset = RSADatasetMultiPP(
                    args,
                    key_dir=args.key_dir, 
                    npz_dir=args.npz_dir, 
                    key_split='fit',
                    npz_split_list=['1_fit', '2_fit', '3_fit', '4_fit']
                )

    val_dataset = RSADatasetMultiPP(
                    args,
                    key_dir=args.key_dir, 
                    npz_dir=args.npz_dir, 
                    key_split='val',
                    npz_split_list=['1_val']
                )

    fit_loader = loader.get_loader(fit_dataset)
    val_loader = loader.get_loader(val_dataset, False)

    from PMI import PMI
    engine = PMI(args, logger)

    for i in range(engine.epoch, args.num_epoch):
        engine.fit(fit_loader)
        if i % args.test_freq == 0:
            engine.validate(val_loader)
            engine.save_model((args.ckpt_dir + '%03d.pth') % (i + 1))
    engine.save_model((args.ckpt_dir + 'final.pth'))

