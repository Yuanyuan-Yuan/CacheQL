import os
import time
from tqdm import tqdm
import numpy as np
import re

sys.path.insert(0, '..')
import config


root_dir = config.key_dir
npz_dir = os.path.join(config.key_dir, 'npz')
key_dir = config.openssl_rsa_key_dir
libg_dir = config.libgcrypt_rsa_key_dir

n_bits = 1024

def extract(input_txt: str, name: str):
    lines = input_txt.split('\n')
    input_txt = ''
    for l in lines:
        if l.startswith(name):
            input_txt += l + '\n'
        elif len(input_txt) > 0 and l.startswith('    '):
            input_txt += l + '\n'
        elif len(input_txt) > 0 and not l.startswith('    '):
            break
        
    mat = re.search(name+':[\(\)0-9a-f:\sx]*', input_txt)
    if not mat:
        assert False, "cannot find corresponding value"

    v_str = mat.group()
    if v_str.count(':') == 1:
        v_str = v_str[v_str.find("(")+1:].strip()
        v_str = v_str[:-1].strip()
        return v_str[2:].upper()

    v_str = v_str.replace(name, '')
    v_str = v_str.replace('\n', '')
    v_str = v_str.replace(' ', '')
    v_str = v_str.replace(':', '')
    return v_str.upper()

def get_expression(key_txt: str):
    n = extract(key_txt, 'modulus')
    e = extract(key_txt, 'publicExponent')
    if not e.startswith('0'):
        e = '0' + e
    d = extract(key_txt, 'privateExponent')
    p = extract(key_txt, 'prime1')
    q = extract(key_txt, 'prime2')
    not_used = extract(key_txt, 'exponent1')
    not_used2= extract(key_txt, 'exponent2')
    u = extract(key_txt, 'coefficient')
    if not u.startswith('0'):
        u = '00' + u
    pri_key_txt = '(private-key\n (rsa\n  ({} #{}#)\n  ({} #{}#)\n  ({} #{}#)\n  ({} #{}#)\n  ({} #{}#)\n  ({} #{}#) )\n)\n'.format('n', n, 'e', e, 'd', d, 'p', p, 'q', q, 'u', u)
    return pri_key_txt

def text_to_libgcrypt(text_path, libg_path):
    with open(text_path, 'r') as f1:
        txt = f1.read()
        s_exp = get_expression(txt)
    with open(libg_path, 'w') as f2:
        f2.write(s_exp)

def text_to_npz(text_path, npz_path):
    start = 'privateExponent'
    end = 'prime1'
    units_len = int(n_bits) // 8
    arr = []
    hex_units = []
    with open(text_path, 'r') as f:
        lines = f.readlines()
        read = False
        for l in lines:
            if end in l:
                break
            if read:
                hex_list = l[:-1].split(':') # delete `\n`
                # print(hex_list)
                for hex_str in hex_list:
                    if len(hex_str):
                        hex_units.append(hex_str)
            elif start in l:
                read = True

        if len(hex_units) < units_len:
            # print('PAD')
            hex_units = ['00'] * (units_len - len(hex_units)) + hex_units
        for hex_str in hex_units[-units_len:]:
            bin_size = 8
            bin_str = (bin(int(hex_str, 16))[2:]).zfill(bin_size)
            arr += [int(s) for s in bin_str]
        # print(len(arr))
        assert len(arr) == int(n_bits)
    np.savez_compressed(npz_path, np.array(arr))

total_num = 6

split2end = {
    'fit': 40000,
    'val': 40000 + 10000
}

split2start = {
    'fit': 0,
    'val': split2end['fit'] 
}

for split in ['fit', 'val']:
    os.makedirs(os.path.join(key_dir, split), exist_ok=True)
    os.makedirs(os.path.join(npz_dir, split), exist_ok=True)
    os.makedirs(os.path.join(libg_dir, split), exist_ok=True)
    idx_list = list(range(split2start[split], split2end[split]))    
    for i in tqdm(idx_list):
        text_path = os.path.join(root_dir, 'cleartext.txt')
        key_path = os.path.join(key_dir, '%s/key_%05d.pem' % (split, i))
        pubkey_path = os.path.join(key_dir, '%s/pubkey_%05d.pem' % (split, i))
        npz_path = os.path.join(npz_dir, '%s/%05d.npz' % (split, i))
        libg_path = os.path.join(libg_dir, '%s/%05d.txt' % (split, i))
        os.system('openssl genrsa -out %s %d' % (key_path, n_bits))
        os.system('openssl rsa -in %s -pubout -out %s' % (key_path, pubkey_path))
        os.system('openssl rsa -in %s -text -out %s' % (key_path, text_path))
        text_to_npz(text_path, npz_path)
        text_to_libgcrypt(text_path, libg_path)