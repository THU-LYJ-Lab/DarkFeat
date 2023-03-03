import os
import cv2
import time
import yaml
import torch
import datetime
from tensorboardX import SummaryWriter
import torchvision.transforms as tvf
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nets.l2net import Quad_L2Net
from nets.geom import getK, getWarp, _grid_positions
from nets.loss import make_detector_loss
from nets.score import extract_kpts
from datasets.noise_simulator import NoiseSimulator
from nets.l2net import Quad_L2Net


class SingleTrainerNoRel:
    def __init__(self, config, device, loader, job_name, start_cnt):
        self.config = config
        self.device = device
        self.loader = loader
        
        # tensorboard writer construction
        os.makedirs('./runs/', exist_ok=True)
        if job_name != '':
            self.log_dir = f'runs/{job_name}'
        else:
            self.log_dir = f'runs/{datetime.datetime.now().strftime("%m-%d-%H%M%S")}'

        self.writer = SummaryWriter(self.log_dir)
        with open(f'{self.log_dir}/config.yaml', 'w') as f:
            yaml.dump(config, f)

        if config['network']['input_type'] == 'gray' or config['network']['input_type'] == 'raw-gray':
            self.model = eval(f'{config["network"]["model"]}(inchan=1)').to(device)
        elif config['network']['input_type'] == 'rgb' or config['network']['input_type'] == 'raw-demosaic':
            self.model = eval(f'{config["network"]["model"]}(inchan=3)').to(device)
        elif config['network']['input_type'] == 'raw':
            self.model = eval(f'{config["network"]["model"]}(inchan=4)').to(device)
        else:
            raise NotImplementedError()

        # noise maker
        self.noise_maker = NoiseSimulator(device)

        # load model
        self.cnt = 0
        if start_cnt != 0:
            self.model.load_state_dict(torch.load(f'{self.log_dir}/model_{start_cnt:06d}.pth'))
            self.cnt = start_cnt + 1

        # optimizer and scheduler
        if self.config['training']['optimizer'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                [{'params': self.model.parameters(), 'initial_lr': self.config['training']['lr']}],
                lr=self.config['training']['lr'],
                momentum=self.config['training']['momentum'],
                weight_decay=self.config['training']['weight_decay'],
            )
        elif self.config['training']['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                [{'params': self.model.parameters(), 'initial_lr': self.config['training']['lr']}],
                lr=self.config['training']['lr'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise NotImplementedError()

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['training']['lr_step'],
            gamma=self.config['training']['lr_gamma'],
            last_epoch=start_cnt
        )
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())


    def save(self, iter_num):
        torch.save(self.model.state_dict(), f'{self.log_dir}/model_{iter_num:06d}.pth')

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self):
        self.model.train()
        
        for epoch in range(2):
            for batch_idx, inputs in enumerate(self.loader):
                self.optimizer.zero_grad()
                t = time.time()

                # preprocess and add noise
                img0_ori, noise_img0_ori = self.preprocess_noise_pair(inputs['img0'], self.cnt)
                img1_ori, noise_img1_ori = self.preprocess_noise_pair(inputs['img1'], self.cnt)

                img0 = img0_ori.permute(0, 3, 1, 2).float().to(self.device)
                img1 = img1_ori.permute(0, 3, 1, 2).float().to(self.device)

                if self.config['network']['input_type'] == 'rgb':
                    # 3-channel rgb
                    RGB_mean = [0.485, 0.456, 0.406]
                    RGB_std  = [0.229, 0.224, 0.225]
                    norm_RGB = tvf.Normalize(mean=RGB_mean, std=RGB_std)
                    img0 = norm_RGB(img0)
                    img1 = norm_RGB(img1)
                    noise_img0 = norm_RGB(noise_img0)
                    noise_img1 = norm_RGB(noise_img1)

                elif self.config['network']['input_type'] == 'gray':
                    # 1-channel
                    img0 = torch.mean(img0, dim=1, keepdim=True)
                    img1 = torch.mean(img1, dim=1, keepdim=True)
                    noise_img0 = torch.mean(noise_img0, dim=1, keepdim=True)
                    noise_img1 = torch.mean(noise_img1, dim=1, keepdim=True)
                    norm_gray0 = tvf.Normalize(mean=img0.mean(), std=img0.std())
                    norm_gray1 = tvf.Normalize(mean=img1.mean(), std=img1.std())
                    img0 = norm_gray0(img0)
                    img1 = norm_gray1(img1)
                    noise_img0 = norm_gray0(noise_img0)
                    noise_img1 = norm_gray1(noise_img1)

                elif self.config['network']['input_type'] == 'raw':
                    # 4-channel
                    pass

                elif self.config['network']['input_type'] == 'raw-demosaic':
                    # 3-channel
                    pass

                else:
                    raise NotImplementedError()

                desc0, score_map0, _, _ = self.model(img0)
                desc1, score_map1, _, _ = self.model(img1)

                cur_feat_size0 = torch.tensor(score_map0.shape[2:])
                cur_feat_size1 = torch.tensor(score_map1.shape[2:])

                desc0 = desc0.permute(0, 2, 3, 1)
                desc1 = desc1.permute(0, 2, 3, 1)
                score_map0 = score_map0.permute(0, 2, 3, 1)
                score_map1 = score_map1.permute(0, 2, 3, 1)

                r_K0 = getK(inputs['ori_img_size0'], cur_feat_size0, inputs['K0']).to(self.device)
                r_K1 = getK(inputs['ori_img_size1'], cur_feat_size1, inputs['K1']).to(self.device)
                
                pos0 = _grid_positions(
                    cur_feat_size0[0], cur_feat_size0[1], img0.shape[0]).to(self.device)

                pos0, pos1, _ = getWarp(
                    pos0, inputs['rel_pose'].to(self.device), inputs['depth0'].to(self.device),
                    r_K0, inputs['depth1'].to(self.device), r_K1, img0.shape[0])

                det_structured_loss, det_accuracy = make_detector_loss(
                    pos0, pos1, desc0, desc1,
                    score_map0, score_map1, img0.shape[0],
                    self.config['network']['use_corr_n'],
                    self.config['network']['loss_type'],
                    self.config
                )

                total_loss = det_structured_loss
                
                self.writer.add_scalar("acc/normal_acc", det_accuracy, self.cnt)
                self.writer.add_scalar("loss/total_loss", total_loss, self.cnt)
                self.writer.add_scalar("loss/det_loss_normal", det_structured_loss, self.cnt)
                print('iter={},\tloss={:.4f},\tacc={:.4f},\t{:.4f}s/iter'.format(self.cnt, total_loss, det_accuracy, time.time()-t))

                if det_structured_loss != 0:
                    total_loss.backward()
                    self.optimizer.step()
                self.lr_scheduler.step()

                if self.cnt % 100 == 0:
                    indices0, scores0 = extract_kpts(
                        score_map0.permute(0, 3, 1, 2),
                        k=self.config['network']['det']['kpt_n'],
                        score_thld=self.config['network']['det']['score_thld'],
                        nms_size=self.config['network']['det']['nms_size'],
                        eof_size=self.config['network']['det']['eof_size'],
                        edge_thld=self.config['network']['det']['edge_thld']
                    )
                    indices1, scores1 = extract_kpts(
                        score_map1.permute(0, 3, 1, 2),
                        k=self.config['network']['det']['kpt_n'],
                        score_thld=self.config['network']['det']['score_thld'],
                        nms_size=self.config['network']['det']['nms_size'],
                        eof_size=self.config['network']['det']['eof_size'],
                        edge_thld=self.config['network']['det']['edge_thld']
                    )

                    if self.config['network']['input_type'] == 'raw':
                        kpt_img0 = self.showKeyPoints(img0_ori[0][..., :3] * 255., indices0[0])
                        kpt_img1 = self.showKeyPoints(img1_ori[0][..., :3] * 255., indices1[0])
                    else:
                        kpt_img0 = self.showKeyPoints(img0_ori[0] * 255., indices0[0])
                        kpt_img1 = self.showKeyPoints(img1_ori[0] * 255., indices1[0])

                    self.writer.add_image('img0/kpts', kpt_img0, self.cnt, dataformats='HWC')
                    self.writer.add_image('img1/kpts', kpt_img1, self.cnt, dataformats='HWC')
                    self.writer.add_image('img0/score_map', score_map0[0], self.cnt, dataformats='HWC')
                    self.writer.add_image('img1/score_map', score_map1[0], self.cnt, dataformats='HWC')

                if self.cnt % 10000 == 0:
                    self.save(self.cnt)
                
                self.cnt += 1


    def showKeyPoints(self, img, indices):
        key_points = cv2.KeyPoint_convert(indices.cpu().float().numpy()[:, ::-1])
        img = img.numpy().astype('uint8')
        img = cv2.drawKeypoints(img, key_points, None, color=(0, 255, 0))
        return img


    def preprocess(self, img, iter_idx):
        if not self.config['network']['noise'] and 'raw' not in self.config['network']['input_type']:
            return img

        raw = self.noise_maker.rgb2raw(img, batched=True)

        if self.config['network']['noise']:
            ratio_dec = min(self.config['network']['noise_maxstep'], iter_idx) / self.config['network']['noise_maxstep']
            raw = self.noise_maker.raw2noisyRaw(raw, ratio_dec=ratio_dec, batched=True)

        if self.config['network']['input_type'] == 'raw':
            return torch.tensor(self.noise_maker.raw2packedRaw(raw, batched=True))

        if self.config['network']['input_type'] == 'raw-demosaic':
            return torch.tensor(self.noise_maker.raw2demosaicRaw(raw, batched=True))

        rgb = self.noise_maker.raw2rgb(raw, batched=True)
        if self.config['network']['input_type'] == 'rgb' or self.config['network']['input_type'] == 'gray':
            return torch.tensor(rgb)

        raise NotImplementedError()


    def preprocess_noise_pair(self, img, iter_idx):
        assert self.config['network']['noise']

        raw = self.noise_maker.rgb2raw(img, batched=True)

        ratio_dec = min(self.config['network']['noise_maxstep'], iter_idx) / self.config['network']['noise_maxstep']
        noise_raw = self.noise_maker.raw2noisyRaw(raw, ratio_dec=ratio_dec, batched=True)

        if self.config['network']['input_type'] == 'raw':
            return torch.tensor(self.noise_maker.raw2packedRaw(raw, batched=True)), \
                   torch.tensor(self.noise_maker.raw2packedRaw(noise_raw, batched=True))

        if self.config['network']['input_type'] == 'raw-demosaic':
            return torch.tensor(self.noise_maker.raw2demosaicRaw(raw, batched=True)), \
                   torch.tensor(self.noise_maker.raw2demosaicRaw(noise_raw, batched=True))

        noise_rgb = self.noise_maker.raw2rgb(noise_raw, batched=True)
        if self.config['network']['input_type'] == 'rgb' or self.config['network']['input_type'] == 'gray':
            return img, torch.tensor(noise_rgb)

        raise NotImplementedError()
