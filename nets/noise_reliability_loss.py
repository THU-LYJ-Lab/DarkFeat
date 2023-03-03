import torch
import torch.nn as nn
from .reliability_loss import APLoss


class MultiPixelAPLoss (nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """
    def __init__(self, sampler, nq=20):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.sampler = sampler
        self.base = 0.25
        self.dec_base = 0.20

    def loss_from_ap(self, ap, rel, noise_ap, noise_rel):
        dec_ap = torch.clamp(ap - noise_ap, min=0, max=1)
        return (1 - ap*noise_rel - (1-noise_rel)*self.base), (1. - dec_ap*(1-noise_rel) - noise_rel*self.dec_base)

    def forward(self, feat0, feat1, noise_feat0, noise_feat1, conf0, conf1, noise_conf0, noise_conf1, pos0, pos1, B, H, W, N=1500):
        # subsample things
        scores, noise_scores, gt, msk, qconf, noise_qconf = self.sampler(feat0, feat1, noise_feat0, noise_feat1, \
            conf0, conf1, noise_conf0, noise_conf1, pos0, pos1, B, H, W, N=1500)
        
        # compute pixel-wise AP
        n = qconf.numel()
        if n == 0: return 0, 0
        scores, noise_scores, gt = scores.view(n,-1), noise_scores, gt.view(n,-1)
        ap = self.aploss(scores, gt).view(msk.shape)
        noise_ap = self.aploss(noise_scores, gt).view(msk.shape)

        pixel_loss = self.loss_from_ap(ap, qconf, noise_ap, noise_qconf)
        
        loss = pixel_loss[0][msk].mean(), pixel_loss[1][msk].mean()
        return loss