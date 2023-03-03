import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from glob import glob
from tqdm import tqdm
import rawpy
import colour_demosaicing

from .InvISP.model.model import InvISPNet
from .utils.common import Notify
from datasets.noise import camera_params, addGStarNoise, addPStarNoise, addQuantNoise, addRowNoise, sampleK


class NoiseSimulator:
    def __init__(self, device, ckpt_path='./datasets/InvISP/pretrained/canon.pth'):
        self.device = device

        # load Invertible ISP Network
        self.net = InvISPNet(channel_in=3, channel_out=3, block_num=8).to(self.device).eval()
        self.net.load_state_dict(torch.load(ckpt_path), strict=False)
        print(Notify.INFO, "Loaded ISPNet checkpoint: {}".format(ckpt_path), Notify.ENDC)

        # white balance parameters
        self.wb = np.array([2020.0, 1024.0, 1458.0, 1024.0])

        # use Canon EOS 5D4 noise parameters provided by ELD
        self.camera_params = camera_params

        # random specify exposure time ratio from 50 to 150
        self.ratio_min = 50
        self.ratio_max = 150
        pass

    # inverse demosaic
    # input: [H, W, 3]
    # output: [H, W]
    def invDemosaic(self, img):
        img_R = img[::2, ::2, 0]
        img_G1 = img[::2, 1::2, 1]
        img_G2 = img[1::2, ::2, 1]
        img_B = img[1::2, 1::2, 2]
        raw_img = np.ones(img.shape[:2])
        raw_img[::2, ::2] = img_R
        raw_img[::2, 1::2] = img_G1
        raw_img[1::2, ::2] = img_G2
        raw_img[1::2, 1::2] = img_B
        return raw_img

    # demosaic - nearest ver
    # input: [H, W]
    # output: [H, W, 3]
    def demosaicNearest(self, img):
        raw = np.ones((img.shape[0], img.shape[1], 3))
        raw[::2, ::2, 0] = img[::2, ::2]
        raw[::2, 1::2, 0] = img[::2, ::2]
        raw[1::2, ::2, 0] = img[::2, ::2]
        raw[1::2, 1::2, 0] = img[::2, ::2]
        raw[::2, ::2, 2] = img[1::2, 1::2]
        raw[::2, 1::2, 2] = img[1::2, 1::2]
        raw[1::2, ::2, 2] = img[1::2, 1::2]
        raw[1::2, 1::2, 2] = img[1::2, 1::2]
        raw[::2, ::2, 1] = img[::2, 1::2]
        raw[::2, 1::2, 1] = img[::2, 1::2]
        raw[1::2, ::2, 1] = img[1::2, ::2]
        raw[1::2, 1::2, 1] = img[1::2, ::2]
        return raw

    # demosaic
    # input: [H, W]
    # output: [H, W, 3]
    def demosaic(self, img):
        return colour_demosaicing.demosaicing_CFA_Bayer_bilinear(img, 'RGGB')

    # load rgb image
    def path2rgb(self, path):
        return torch.from_numpy(np.array(PILImage.open(path))/255.0)

    # InvISP
    # input: rgb image [H, W, 3]
    # output: raw image [H, W]
    def rgb2raw(self, rgb, batched=False):
        # 1. rgb -> invnet
        if not batched:
            rgb = rgb.unsqueeze(0)

        rgb = rgb.permute(0,3,1,2).float().to(self.device)
        with torch.no_grad():
            reconstruct_raw = self.net(rgb, rev=True)

        pred_raw = reconstruct_raw.detach().permute(0,2,3,1)
        pred_raw = torch.clamp(pred_raw, 0, 1)

        if not batched:
            pred_raw = pred_raw[0, ...]
            
        pred_raw = pred_raw.cpu().numpy()

        # 2. -> inv gamma
        norm_value = np.power(16383, 1/2.2)
        pred_raw *= norm_value          
        pred_raw = np.power(pred_raw, 2.2)

        # 3. -> inv white balance
        wb = self.wb / self.wb.max()
        pred_raw = pred_raw / wb[:-1]

        # 4. -> add black level
        pred_raw += self.camera_params['black_level']

        # 5. -> inv demosaic
        if not batched:
            pred_raw = self.invDemosaic(pred_raw)
        else:
            preds = []
            for i in range(pred_raw.shape[0]):
                preds.append(self.invDemosaic(pred_raw[i]))
            pred_raw = np.stack(preds, axis=0)

        return pred_raw

    
    def raw2noisyRaw(self, raw, ratio_dec=1, batched=False):
        if not batched:
            ratio = (random.uniform(self.ratio_min, self.ratio_max) - 1) * ratio_dec + 1
            raw = raw.copy() / ratio

            K = sampleK(self.camera_params['Kmin'], self.camera_params['Kmax'])
            q = 1 / (self.camera_params['max_value'] - self.camera_params['black_level'])

            raw = addPStarNoise(raw, K)
            raw = addGStarNoise(raw, K, self.camera_params['G_shape'], self.camera_params['Profile-1']['G_scale'])
            raw = addRowNoise(raw, K, self.camera_params['Profile-1']['R_scale'])
            raw = addQuantNoise(raw, q)
            raw *= ratio
            return raw

        else:
            raw = raw.copy()
            for i in range(raw.shape[0]):
                ratio = random.uniform(self.ratio_min, self.ratio_max)
                raw[i] /= ratio

                K = sampleK(self.camera_params['Kmin'], self.camera_params['Kmax'])
                q = 1 / (self.camera_params['max_value'] - self.camera_params['black_level'])

                raw[i] = addPStarNoise(raw[i], K)
                raw[i] = addGStarNoise(raw[i], K, self.camera_params['G_shape'], self.camera_params['Profile-1']['G_scale'])
                raw[i] = addRowNoise(raw[i], K, self.camera_params['Profile-1']['R_scale'])
                raw[i] = addQuantNoise(raw[i], q)
                raw[i] *= ratio
            return raw

    def raw2rgb(self, raw, batched=False):
        # 1. -> demosaic
        if not batched:
            raw = self.demosaic(raw)
        else:
            raws = []
            for i in range(raw.shape[0]):
                raws.append(self.demosaic(raw[i]))
            raw = np.stack(raws, axis=0)

        # 2. -> substract black level
        raw -= self.camera_params['black_level']
        raw = np.clip(raw, 0, self.camera_params['max_value'] - self.camera_params['black_level'])

        # 3. -> white balance
        wb = self.wb / self.wb.max()
        raw = raw * wb[:-1]

        # 4. -> gamma
        norm_value = np.power(16383, 1/2.2)            
        raw = np.power(raw, 1/2.2)
        raw /= norm_value

        # 5. -> ispnet
        if not batched:
            input_raw_img = torch.Tensor(raw).permute(2,0,1).float().to(self.device)[np.newaxis, ...]
        else:
            input_raw_img = torch.Tensor(raw).permute(0,3,1,2).float().to(self.device)

        with torch.no_grad():
            reconstruct_rgb = self.net(input_raw_img)
            reconstruct_rgb = torch.clamp(reconstruct_rgb, 0, 1)

        pred_rgb = reconstruct_rgb.detach().permute(0,2,3,1)

        if not batched:
            pred_rgb = pred_rgb[0, ...]
        pred_rgb = pred_rgb.cpu().numpy()

        return pred_rgb


    def raw2packedRaw(self, raw, batched=False):
        # 1. -> substract black level
        raw -= self.camera_params['black_level']
        raw = np.clip(raw, 0, self.camera_params['max_value'] - self.camera_params['black_level'])
        raw /= self.camera_params['max_value']

        # 2. pack
        if not batched:
            im = np.expand_dims(raw, axis=2)
            img_shape = im.shape
            H = img_shape[0]
            W = img_shape[1]

            out = np.concatenate((im[0:H:2, 0:W:2, :],
                                  im[0:H:2, 1:W:2, :],
                                  im[1:H:2, 1:W:2, :],
                                  im[1:H:2, 0:W:2, :]), axis=2)
        else:
            im = np.expand_dims(raw, axis=3)
            img_shape = im.shape
            H = img_shape[1]
            W = img_shape[2]

            out = np.concatenate((im[:, 0:H:2, 0:W:2, :],
                                  im[:, 0:H:2, 1:W:2, :],
                                  im[:, 1:H:2, 1:W:2, :],
                                  im[:, 1:H:2, 0:W:2, :]), axis=3)
        return out

    def raw2demosaicRaw(self, raw, batched=False):
        # 1. -> demosaic
        if not batched:
            raw = self.demosaic(raw)
        else:
            raws = []
            for i in range(raw.shape[0]):
                raws.append(self.demosaic(raw[i]))
            raw = np.stack(raws, axis=0)

        # 2. -> substract black level
        raw -= self.camera_params['black_level']
        raw = np.clip(raw, 0, self.camera_params['max_value'] - self.camera_params['black_level'])
        raw /= self.camera_params['max_value']
        return raw
