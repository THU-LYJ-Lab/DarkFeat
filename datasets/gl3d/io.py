import os
import re
import cv2
import numpy as np

from ..utils.common import Notify

def read_list(list_path):
    """Read list."""
    if list_path is None or not os.path.exists(list_path):
        print(Notify.FAIL, 'Not exist', list_path, Notify.ENDC)
        exit(-1)
    content = open(list_path).read().splitlines()
    return content


def load_pfm(pfm_path):
    with open(pfm_path, 'rb') as fin:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = str(fin.readline().decode('UTF-8')).rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$',
                             fin.readline().decode('UTF-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float((fin.readline().decode('UTF-8')).rstrip())
        if scale < 0:  # little-endian
            data_type = '<f'
        else:
            data_type = '>f'  # big-endian
        data_string = fin.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flip(data, 0)
    return data


def _parse_img(img_paths, idx, config):
    img_path = img_paths[idx]
    img = cv2.imread(img_path)[:, :, ::-1]
    if config['resize'] > 0:
        img = cv2.resize(
            img, (config['resize'], config['resize']))
    return img


def _parse_depth(depth_paths, idx, config):
    depth = load_pfm(depth_paths[idx])

    if config['resize'] > 0:
        target_size = config['resize']
    if config['input_type'] == 'raw':
        depth = cv2.resize(depth, (int(target_size/2), int(target_size/2)))
    else:
        depth = cv2.resize(depth, (target_size, target_size))
    return depth


def _parse_kpts(kpts_paths, idx, config):
    kpts = np.load(kpts_paths[idx])['pts']
    # output: [N, 2] (W first H last)
    return kpts
