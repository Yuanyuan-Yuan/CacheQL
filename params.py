import os
import json
import torch
import argparse

import config

class Params():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ID', type=int, default=-1)
        parser.add_argument('--exp_name', type=str, default='test')
        # parser.add_argument('--key_dir', type=str, default='/data/yyuanaq/data/RSA-sign-mbedTLS/private_key_npz/')
        # parser.add_argument('--npz_dir', type=str, default='/data/yyuanaq/data/RSA-sign-mbedTLS/npz_cacheline/')
        # parser.add_argument('--output_root', type=str, default='/data/yyuanaq/output/RSA/')

        parser.add_argument('--side', type=str, default='', choices=[
            '', 'cacheline', 'cachebank'
        ])
        parser.add_argument('--software', type=str, default='rsa_openssl_0.9.7c', choices=[
            'rsa_openssl_0.9.7c', 'rsa_openssl_3.0.0',
            'rsa_mbedtls_2.15.0', 'rsa_mbedtls_3.0.0',
            'rsa_sign_libgcrypt_1.6.1', 'rsa_sign_libgcrypt_1.9.4',
            'aes_openssl_0.9.7c', 'aes_openssl_3.0.0',
            'aes_mbedtls_2.15.0', 'aes_mbedtls_3.0.0',
            'libjpeg-turbo-2.1.2'
        ])
        parser.add_argument('--setting', type=str, default='SDA', choices=[
            'SDA', 'SCB', 'pp_dcache', 'pp_icache'
        ])
        
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=4)
        
        parser.add_argument('--lr', type=float, default=2e-4)
        parser.add_argument('--beta1', type=float, default=0.5)
        
        parser.add_argument('--dim', type=int, default=128)
        parser.add_argument('--nc', type=int, default=3)
        parser.add_argument('--size', type=int, default=512)
        parser.add_argument('--hidden_dim', type=int, default=64)
        parser.add_argument('--output_dim', type=int, default=1)
        parser.add_argument('--n_layer', type=int, default=2)

        parser.add_argument('--image_size', type=int, default=128)
        parser.add_argument('--image_nc', type=int, default=3)

        parser.add_argument('--num_epoch', type=int, default=50)
        parser.add_argument('--test_freq', type=int, default=1)

        parser.add_argument('--use_norm', type=int, default=1, choices=[0, 1])
        parser.add_argument('--use_bias', type=int, default=1, choices=[0, 1])
        parser.add_argument('--repeat_num', type=int, default=16)

        parser.add_argument('--use_IG', type=int, default=0, choices=[0, 1])

        self.args = parser.parse_args()
        # self.args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.parser = parser

        self.PADLENGTH = {
            'rsa_openssl_0.9.7c-SDA': (256, 20),
            'rsa_openssl_0.9.7c-SCB': (256, 8),
            'rsa_openssl_3.0.0-SDA': (256, 64),
            'rsa_openssl_3.0.0-SCB': (256, 40),

            'rsa_mbedtls_2.15.0-SDA': (256, 14),
            'rsa_mbedtls_2.15.0-SCB': (256, 6),
            'rsa_mbedtls_3.0.0-SDA': (256, 16),
            'rsa_mbedtls_3.0.0-SCB': (256, 6),

            'rsa_sign_libgcrypt_1.6.1-SDA': (256, 5),
            'rsa_sign_libgcrypt_1.6.1-SCB': (256, 3),
            'rsa_sign_libgcrypt_1.9.4-SDA': (256, 36),
            'rsa_sign_libgcrypt_1.9.4-SCB': (256, 12),

            'aes_openssl_3.0.0-SDA': (32, 1),
            'aes_openssl_0.9.7c-SDA': (32, 1),
            'aes_mbedtls_3.0.0-SDA': (32, 6),
            'aes_mbedtls_2.15.0-SDA': (32, 6),

            'rsa_openssl_3.0.0-pp_dcache': (256, 2),
            'rsa_openssl_3.0.0-pp_icache': (128, 4),
            'rsa_openssl_0.9.7c-pp_dcache': (256, 2),
            'rsa_openssl_0.9.7c-pp_icache': (128, 4),
        
            'libjpeg-turbo-2.1.2-SDA': (256, 4),
            'libjpeg-turbo-2.1.2-SCB': (128, 6),
        }

        self.DET_PADLENGTH = {
            'rsa_openssl_0.9.7c-memory': (256, 14),
            'rsa_openssl_0.9.7c-branch': (256, 5),

            'rsa_openssl_3.0.0-memory': (256, 40),
            'rsa_openssl_3.0.0-branch': (256, 24),

            'rsa_sign_libgcrypt_1.9.4-memory': (256, 26),
            'rsa_sign_libgcrypt_1.9.4-branch': (256, 10),
        }


        self.DEC_PADLENGTH = {
            'rsa_openssl_0.9.7c-SDA': (256, 14),
            'rsa_openssl_0.9.7c-SCB': (256, 5),
            'rsa_openssl_3.0.0-SDA': (256, 15),
            'rsa_openssl_3.0.0-SCB': (256, 10),

            'rsa_mbedtls_2.15.0-SDA': (256, 2),
            'rsa_mbedtls_2.15.0-SCB': (256, 1),
            'rsa_mbedtls_3.0.0-SDA': (256, 2),
            'rsa_mbedtls_3.0.0-SCB': (256, 1),

            'rsa_sign_libgcrypt_1.6.1-SDA': (128, 19),
            'rsa_sign_libgcrypt_1.6.1-SCB': (64, 33), # (128, 8)
            'rsa_sign_libgcrypt_1.9.4-SDA': (256, 22),
            'rsa_sign_libgcrypt_1.9.4-SCB': (256, 8),
        }

        self.DEC_DET_PADLENGTH = {
            'rsa_openssl_0.9.7c-SDA': (256, 13),
            'rsa_openssl_0.9.7c-SCB': (256, 4),

            'rsa_openssl_3.0.0-SDA': (256, 9),
            'rsa_openssl_3.0.0-SCB': (256, 4),

            'rsa_sign_libgcrypt_1.9.4-SDA': (256, 25),
            'rsa_sign_libgcrypt_1.9.4-SCB': (256, 9),

        }

        self.DEC_START = {
            'rsa_openssl_0.9.7c-SDA': 256 * 256 * 6,
            'rsa_openssl_0.9.7c-SCB': 256 * 256 * 3,
            'rsa_openssl_3.0.0-SDA': 256 * 256 * 49,
            'rsa_openssl_3.0.0-SCB': 256 * 256 * 30,

            'rsa_mbedtls_2.15.0-SDA': 256 * 256 * 12,
            'rsa_mbedtls_2.15.0-SCB': 256 * 256 * 5,
            'rsa_mbedtls_3.0.0-SDA': 256 * 256 * 14,
            'rsa_mbedtls_3.0.0-SCB': 256 * 256 * 5,

            'rsa_sign_libgcrypt_1.6.1-SDA': 128 * 128 * 1,
            'rsa_sign_libgcrypt_1.6.1-SCB': 64 * 64 * 15, # 128 * 128 * 4
            'rsa_sign_libgcrypt_1.9.4-SDA': 256 * 256 * 14,
            'rsa_sign_libgcrypt_1.9.4-SCB': 256 * 256 * 4,
        }

        self.DEC_DET_START = {

            'rsa_openssl_0.9.7c-SDA': 256 * 256 * 1,
            'rsa_openssl_0.9.7c-SCB': 256 * 256 * 1,

            'rsa_openssl_3.0.0-SDA': 256 * 256 * 31,
            'rsa_openssl_3.0.0-SCB': 256 * 256 * 20,

            'rsa_sign_libgcrypt_1.9.4-SDA': 256 * 256 * 1,
            'rsa_sign_libgcrypt_1.9.4-SCB': 256 * 256 * 1,
        }

    def parse(self):
        if 'pp_' in self.args.setting:
            self.args.use_norm = 0
            self.args.use_bias = 0
        else:
            self.args.use_norm = 1
            self.args.use_bias = 1

        if 'openssl' in self.args.software:
            if 'rsa' in self.args.software:
                self.args.key_dir = config.openssl_rsa_key_dir
            else:
                self.args.key_dir = config.openssl_aes_key_dir
        elif 'libgcrypt' in self.args.software:
            self.args.key_dir = config.libgcrypt_rsa_key_dir

        self.args.key_length = 1024 if 'rsa' in self.args.software else 128

        return self.args

    def save_params(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.args.__dict__, f)

    def print_options(self, params):
        message = ''
        message += '----------------- Params ---------------\n'
        for k, v in sorted(vars(params).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

if __name__ == '__main__':
    p = Params()
    args = p.parse()
    args.batch_size = 1
    p.print_options(args)