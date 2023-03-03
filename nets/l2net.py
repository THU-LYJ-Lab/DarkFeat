import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .score import peakiness_score


class BaseNet(nn.Module):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """
    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        super(BaseNet, self).__init__()
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, k_pool = 1, pool_type='max', bias=False):
        # as in the original implementation, dilation is applied at the end of layer, so it will have impact only from next layer
        d = self.dilation * dilation
        # if self.dilated: 
        #     conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
        #     self.dilation *= stride
        # else:
        #     conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride, bias=bias)

        ops = nn.ModuleList([])

        ops.append( nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params) )
        if bn and self.bn: ops.append( self._make_bn(outd) )
        if relu: ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd
        
        if k_pool > 1:
            if pool_type == 'avg':
                ops.append(torch.nn.AvgPool2d(kernel_size=k_pool))
            elif pool_type == 'max':
                ops.append(torch.nn.MaxPool2d(kernel_size=k_pool))
            else:
                print(f"Error, unknown pooling type {pool_type}...")

        return nn.Sequential(*ops)


class Quad_L2Net(BaseNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """
    def __init__(self, dim=128, mchan=4, relu22=False, **kw):
        BaseNet.__init__(self, **kw)
        self.conv0 = self._add_conv(  8*mchan)
        self.conv1 = self._add_conv(  8*mchan, bn=False)
        self.bn1 = self._make_bn(8*mchan)
        self.conv2 = self._add_conv( 16*mchan, stride=2)
        self.conv3 = self._add_conv( 16*mchan, bn=False)
        self.bn3 = self._make_bn(16*mchan)
        self.conv4 = self._add_conv( 32*mchan, stride=2)
        self.conv5 = self._add_conv( 32*mchan)
        # replace last 8x8 convolution with 3 3x3 convolutions
        self.conv6_0 = self._add_conv( 32*mchan)
        self.conv6_1 = self._add_conv( 32*mchan)
        self.conv6_2 = self._add_conv(dim, bn=False, relu=False)
        self.out_dim = dim

        self.moving_avg_params = nn.ParameterList([
            Parameter(torch.tensor(1.), requires_grad=False),
            Parameter(torch.tensor(1.), requires_grad=False),
            Parameter(torch.tensor(1.), requires_grad=False)
        ])

    def forward(self, x):
        # x: [N, C, H, W]
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x1_bn = self.bn1(x1)
        x2 = self.conv2(x1_bn)
        x3 = self.conv3(x2)
        x3_bn = self.bn3(x3)
        x4 = self.conv4(x3_bn)
        x5 = self.conv5(x4)
        x6_0 = self.conv6_0(x5)
        x6_1 = self.conv6_1(x6_0)
        x6_2 = self.conv6_2(x6_1)

        # calculate score map
        comb_weights = torch.tensor([1., 2., 3.], device=x.device)
        comb_weights /= torch.sum(comb_weights)
        ksize = [3, 2, 1]
        det_score_maps = []

        for idx, xx in enumerate([x1, x3, x6_2]):
            if self.training:
                instance_max = torch.max(xx)
                self.moving_avg_params[idx].data = self.moving_avg_params[idx] * 0.99 + instance_max.detach() * 0.01
            else:
                pass

            alpha, beta = peakiness_score(xx, self.moving_avg_params[idx].detach(), ksize=3, dilation=ksize[idx])

            score_vol = alpha * beta
            det_score_map = torch.max(score_vol, dim=1, keepdim=True)[0]
            det_score_map = F.interpolate(det_score_map, size=x.shape[2:], mode='bilinear', align_corners=True)
            det_score_map = comb_weights[idx] * det_score_map
            det_score_maps.append(det_score_map)

        det_score_map = torch.sum(torch.stack(det_score_maps, dim=0), dim=0)
        # print([param.data for param in self.moving_avg_params])

        return x6_2, det_score_map, x1, x3
