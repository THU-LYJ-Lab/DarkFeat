import glob
import rawpy
import cv2
import os
import numpy as np
import colour_demosaicing
from tqdm import tqdm


def process_raw(args, path, w_new, h_new):
    raw = rawpy.imread(str(path)).raw_image_visible
    if '_00200_' in str(path) or '_00100_' in str(path):
        raw = np.clip(raw.astype('float32') - 512, 0, 65535)
    else:
        raw = np.clip(raw.astype('float32') - 2048, 0, 65535)
    img = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, 'RGGB').astype('float32')
    img = np.clip(img, 0, 16383)

    # HistEQ start
    if args.histeq:
        img2 = np.zeros_like(img)
        for i in range(3):
            hist,bins = np.histogram(img[..., i].flatten(),16384,[0,16384])
            cdf = hist.cumsum()
            cdf_normalized = cdf * float(hist.max()) / cdf.max()
            cdf_m = np.ma.masked_equal(cdf,0)
            cdf_m = (cdf_m - cdf_m.min())*16383/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint16')
            img2[..., i] = cdf[img[..., i].astype('int16')]
            img[..., i] = img2[..., i].astype('float32')
    # HistEQ end

    m = img.mean()
    d = np.abs(img - img.mean()).mean()
    img = (img - m + 2*d) / 4/d * 255
    image = np.clip(img, 0, 255)

    image = cv2.resize(image.astype('float32'), (w_new, h_new), interpolation=cv2.INTER_AREA)

    if args.histeq:
        path=str(path)
        os.makedirs('/'.join(path.split('/')[:-2]+[path.split('/')[-2]+'-npy']), exist_ok=True)
        np.save('/'.join(path.split('/')[:-2]+[path.split('/')[-2]+'-npy']+[path.split('/')[-1].replace('cr2','npy')]), image)
    else:
        path=str(path)
        os.makedirs('/'.join(path.split('/')[:-2]+[path.split('/')[-2]+'-npy-nohisteq']), exist_ok=True)
        np.save('/'.join(path.split('/')[:-2]+[path.split('/')[-2]+'-npy-nohisteq']+[path.split('/')[-1].replace('cr2','npy')]), image)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=int(640))
    parser.add_argument('--W', type=int, default=int(960))
    parser.add_argument('--histeq', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='/data/hyz/MID/')
    args = parser.parse_args()

    path_ls = glob.glob(args.dataset_dir + '/*/pair*/?????/*')
    for path in tqdm(path_ls):
        process_raw(args, path, args.W, args.H)

