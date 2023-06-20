import os
import time
import subprocess
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '..')
import config

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

cpu_id = int(sys.argv[1])
seg_id = int(sys.argv[2])
total_num = 16
assert seg_id in list(range(total_num))

SOFTWARE = 'libjpeg-turbo-2.1.2'

N_SET = 64 # number of cache set
REPEAT_NUM = 8
SETTING = 'pp_dcache'

out_path = ('image_%03d-%03d.bmp' % (cpu_id, seg_id))
log_path = ('%s_%s_%03d-%03d.txt' % (SOFTWARE, SETTING, cpu_id, seg_id))

pp_cmd = ('taskset -c %d %s %s' % (cpu_id, os.path.join(config.Mastik_dir, '/demo/L1-d-cache'), log_path))
# replace `L1-d-cache` with `L1-i-cache` for L1 I cache

PAD_LEN = 128 * REPEAT_NUM

make_path(os.path.join(config.trace_dir, SETTING))
make_path(os.path.join(config.trace_dir, SETTING, SOFTWARE))
npz_dir = os.path.join(config.trace_dir, SETTING, SOFTWARE, 'npz')
make_path(npz_dir)

input_dir = config.image_dir

sub_list = ['fit']

SUBNUM_DICT = {
    'fit': [1, 2, 3, 4],
    'val': [1]
}

for sub in sub_list:
    for repeat_idx in SUBNUM_DICT[sub]:
        make_path(os.path.join(npz_dir, str(repeat_idx) + '_' + sub))

        total_image_list = sorted(os.listdir(input_dir + sub))
        unit_len = int(len(total_image_list) // total_num)

        if seg_id == total_num - 1:
            image_list = total_image_list[seg_id*unit_len:]
        else:
            image_list = total_image_list[seg_id*unit_len:(seg_id+1)*unit_len]

        for i, image_name in enumerate(tqdm(image_list)):            
            in_path = os.path.join(input_dir, sub, image_name)
            prefix = image_name.split('.')[0]
            suffix = '.npz'
            npz_path = os.path.join(npz_dir, str(repeat_idx) + '_' + sub, prefix + suffix)
            
            command = '%s %s %s' % (os.path.join(config.libjpeg_dir, 'tjexample'), in_path, out_path)
            victim_cmd = ('taskset -c %d %s' % (cpu_id, command))
            
            data_list = []
            for try_idx in range(REPEAT_NUM):
                pp_proc = subprocess.Popen([pp_cmd], shell=True)
                time.sleep(0.002)
                start_time = time.time()
                os.system(victim_cmd)
                end_time = time.time()
                time.sleep(0.004)
                os.system("sudo kill -9 " + str(pp_proc.pid))

                with open(log_path, 'r') as f:
                    lines = f.readlines()

                access_list = []
                for l in lines:
                    if len(l) > 1:
                        info = l.strip().split(' ')
                        cur = float(info[0])
                        if cur > end_time:
                            break
                        if cur >= start_time:
                            access_list.append(np.array(list(map(int, info[1:]))))

                for i in range(len(access_list) - 1):
                    prev = access_list[i]
                    nxt = access_list[i + 1]
                    try:
                        miss = (prev == 1) & (nxt == 0)
                        if np.sum(miss) > 0:
                            data_list.append(miss.astype(float))
                    except:
                        print('mismatch.') 
            if len(data_list) < PAD_LEN:
                for _ in range(PAD_LEN - len(data_list)):
                    data_list.append(np.zeros(N_SET))
            else:
                data_list = data_list[:PAD_LEN]
            np.savez_compressed(npz_path, np.array(data_list))
