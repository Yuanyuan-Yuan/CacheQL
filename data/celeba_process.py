import os
from tqdm import tqdm
import PIL
from PIL import Image

def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

input_dir = './img_align_celeba/'
output_dir = './celeba_crop128/'

txt_path = './celeba_partition.txt'


def center_crop(img, new_width=128, new_height=128):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return img.crop((left, top, right, bottom))

split_dic = {}
with open(txt_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        file_name, split = l.strip().split(' ')
        if file_name in split_dic.keys():
            print('Error.')
        split_dic[file_name] = int(split)

sub_list = ['fit/', 'val/', 'test/']

make_path(output_dir)
for sub in sub_list:
    make_path(output_dir + sub)

file_list = sorted(os.listdir(input_dir))
for i, file_name in enumerate(tqdm(file_list)):
    input_path = input_dir + file_name
    split_idx = split_dic[file_name]
    output_path = output_dir + sub_list[split_idx] + file_name
    img = Image.open(input_path)
    out_img = center_crop(
            img=img,
            new_width=128,
            new_height=128
        )
    out_img.save(output_path)