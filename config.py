import os
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Mastik_dir = '/MASTIK'
# update this path after downloading the Mastik project


openssl_rsa_key_dir = '/export/d3/user/dataset/crypto-lib/key/openssl/key/'
# openssl and mbedtls RSA
openssl_aes_key_dir = '/export/d3/user/dataset/crypto-lib/key/aes/key/'

key_dir = './data/key/'
openssl_rsa_key_dir = os.path.join(key_dir, 'openssl-rsa')
mbedtls_rsa_key_dir = openssl_rsa_key_dir
# Note that MbedTLs supports directly use OpenSSL keys

libgcrypt_rsa_key_dir = os.path.join(key_dir, 'libgcrypt-rsa')
openssl_aes_key_dir = os.path.join(key_dir, 'openssl-aes')

image_dir = './data/celeba_crop128/'
trace_dir = './data/trace/npz/'
inst_dir = '/data/trace/raw/'

output_dir = './output'

lib_dir = './software/testcases_nopie/'
libjpeg_dir = './software/libjpeg-turbo-build/'
