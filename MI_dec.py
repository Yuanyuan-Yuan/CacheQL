

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
    
    from dataset import *
    from params import Params

    args = Params().parse()

    if len(args.side) > 0:
        args.exp_name = 'dec-mi-%s-%s-%s' % (args.software, args.setting, args.side)
    else:
        args.exp_name = 'dec-mi-%s-%s' % (args.software, args.setting)
    print(args.exp_name)

    args.npz_dir = os.path.join(config.trace_dec_dir, args.setting, args.software)
    
    if 'rsa' in args.software:
        args.key_dir = config.rsa_key_npz_dir
    elif 'aes' in args.software:
        args.key_dir = config.aes_key_npz_dir
    
    args.start_index = args.DEC_START['%s-%s' % (args.software, args.setting)]
    (args.size, args.nc) = args.DEC_PADLENGTH['%s-%s' % (args.software, args.setting)]

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

    fit_dataset = RSADatasetMultiDec(
                    args,
                    key_dir=args.key_dir, 
                    npz_dir=args.npz_dir, 
                    key_split='fit',
                    npz_split_list=['1_fit', '2_fit', '3_fit', '4_fit']
                )

    val_dataset = RSADatasetMultiDec(
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

