import torch
from torch import nn
from torch.nn.parameter import Parameter
import torchvision.transforms as tvf
import torch.nn.functional as F
import numpy as np


def gather_nd(params, indices):
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()


# input: pos [kpt_n, 2]; inputs [H, W, 128] / [H, W]
# output: [kpt_n, 128] / [kpt_n]
def interpolate(pos, inputs, nd=True):
    h = inputs.shape[0]
    w = inputs.shape[1]

    i = pos[:, 0]
    j = pos[:, 1]

    i_top_left = torch.clamp(torch.floor(i).int(), 0, h - 1)
    j_top_left = torch.clamp(torch.floor(j).int(), 0, w - 1)

    i_top_right = torch.clamp(torch.floor(i).int(), 0, h - 1)
    j_top_right = torch.clamp(torch.ceil(j).int(), 0, w - 1)

    i_bottom_left = torch.clamp(torch.ceil(i).int(), 0, h - 1)
    j_bottom_left = torch.clamp(torch.floor(j).int(), 0, w - 1)

    i_bottom_right = torch.clamp(torch.ceil(i).int(), 0, h - 1)
    j_bottom_right = torch.clamp(torch.ceil(j).int(), 0, w - 1)

    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    if nd:
        w_top_left = w_top_left[..., None]
        w_top_right = w_top_right[..., None]
        w_bottom_left = w_bottom_left[..., None]
        w_bottom_right = w_bottom_right[..., None]

    interpolated_val = (
        w_top_left * gather_nd(inputs, torch.stack([i_top_left, j_top_left], axis=-1)) +
        w_top_right * gather_nd(inputs, torch.stack([i_top_right, j_top_right], axis=-1)) +
        w_bottom_left * gather_nd(inputs, torch.stack([i_bottom_left, j_bottom_left], axis=-1)) +
        w_bottom_right *
        gather_nd(inputs, torch.stack([i_bottom_right, j_bottom_right], axis=-1))
    )

    return interpolated_val


def edge_mask(inputs, n_channel, dilation=1, edge_thld=5):
    b, c, h, w = inputs.size()
    device = inputs.device

    dii_filter = torch.tensor(
        [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
    ).view(1, 1, 3, 3)
    dij_filter = 0.25 * torch.tensor(
        [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
    ).view(1, 1, 3, 3)
    djj_filter = torch.tensor(
        [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
    ).view(1, 1, 3, 3)

    dii = F.conv2d(
        inputs.view(-1, 1, h, w), dii_filter.to(device), padding=dilation, dilation=dilation
    ).view(b, c, h, w)
    dij = F.conv2d(
        inputs.view(-1, 1, h, w), dij_filter.to(device), padding=dilation, dilation=dilation
    ).view(b, c, h, w)
    djj = F.conv2d(
        inputs.view(-1, 1, h, w), djj_filter.to(device), padding=dilation, dilation=dilation
    ).view(b, c, h, w)

    det = dii * djj - dij * dij
    tr = dii + djj
    del dii, dij, djj

    threshold = (edge_thld + 1) ** 2 / edge_thld
    is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

    return is_not_edge


# input: score_map [batch_size, 1, H, W]
# output: indices [2, k, 2], scores [2, k]
def extract_kpts(score_map, k=256, score_thld=0, edge_thld=0, nms_size=3, eof_size=5):
    h = score_map.shape[2]
    w = score_map.shape[3]

    mask = score_map > score_thld
    if nms_size > 0:
        nms_mask = F.max_pool2d(score_map, kernel_size=nms_size, stride=1, padding=nms_size//2)
        nms_mask = torch.eq(score_map, nms_mask)
        mask = torch.logical_and(nms_mask, mask)
    if eof_size > 0:
        eof_mask = torch.ones((1, 1, h - 2 * eof_size, w - 2 * eof_size), dtype=torch.float32, device=score_map.device)
        eof_mask = F.pad(eof_mask, [eof_size] * 4, value=0)
        eof_mask = eof_mask.bool()
        mask = torch.logical_and(eof_mask, mask)
    if edge_thld > 0:
        non_edge_mask = edge_mask(score_map, 1, dilation=3, edge_thld=edge_thld)
        mask = torch.logical_and(non_edge_mask, mask)

    bs = score_map.shape[0]
    if bs is None:
        indices = torch.nonzero(mask)[0]
        scores = gather_nd(score_map, indices)[0]
        sample = torch.sort(scores, descending=True)[1][0:k]
        indices = indices[sample].unsqueeze(0)
        scores = scores[sample].unsqueeze(0)
    else:
        indices = []
        scores = []
        for i in range(bs):
            tmp_mask = mask[i][0]
            tmp_score_map = score_map[i][0]
            tmp_indices = torch.nonzero(tmp_mask)
            tmp_scores = gather_nd(tmp_score_map, tmp_indices)
            tmp_sample = torch.sort(tmp_scores, descending=True)[1][0:k]
            tmp_indices = tmp_indices[tmp_sample]
            tmp_scores = tmp_scores[tmp_sample]
            indices.append(tmp_indices)
            scores.append(tmp_scores)
        try:
            indices = torch.stack(indices, dim=0)
            scores = torch.stack(scores, dim=0)
        except:
            min_num = np.min([len(i) for i in indices])
            indices = torch.stack([i[:min_num] for i in indices], dim=0)
            scores = torch.stack([i[:min_num] for i in scores], dim=0)
    return indices, scores


# input: [batch_size, C, H, W]
# output: [batch_size, C, H, W], [batch_size, C, H, W]
def peakiness_score(inputs, moving_instance_max, ksize=3, dilation=1):
    inputs = inputs / moving_instance_max
    
    batch_size, C, H, W = inputs.shape

    pad_size = ksize // 2 + (dilation - 1)
    kernel = torch.ones([C, 1, ksize, ksize], device=inputs.device) / (ksize * ksize)
    
    pad_inputs = F.pad(inputs, [pad_size] * 4, mode='reflect')

    avg_spatial_inputs = F.conv2d(
        pad_inputs,
        kernel,
        stride=1,
        dilation=dilation,
        padding=0,
        groups=C
    )
    avg_channel_inputs = torch.mean(inputs, axis=1, keepdim=True) # channel dimension is 1
    # print(avg_spatial_inputs.shape)

    alpha = F.softplus(inputs - avg_spatial_inputs)
    beta = F.softplus(inputs - avg_channel_inputs)

    return alpha, beta


class DarkFeat(nn.Module):
    default_config = {
        'model_path': '',
        'input_type': 'raw-demosaic',
        'kpt_n': 5000,
        'kpt_refinement': True,
        'score_thld': 0.5,
        'edge_thld': 10,
        'multi_scale': False,
        'multi_level': True,
        'nms_size': 3,
        'eof_size': 5,
        'need_norm': True,
        'use_peakiness': True
    }

    def __init__(self, model_path='', inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        super(DarkFeat, self).__init__()
        inchan = 3 if self.default_config['input_type'] == 'rgb' or self.default_config['input_type'] == 'raw-demosaic' else 1
        self.config = {**self.default_config}

        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.config['model_path'] = model_path

        dim = 128
        mchan = 4

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
        self.clf = nn.Conv2d(128, 2, kernel_size=1)

        state_dict = torch.load(self.config["model_path"])
        new_state_dict = {}
        
        for key in state_dict:
            if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
                new_state_dict[key] = state_dict[key]

        self.load_state_dict(new_state_dict)
        print('Loaded DarkFeat model')
        
    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine, track_running_stats=False)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True, k_pool = 1, pool_type='max', bias=False):
        d = self.dilation * dilation
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

    def forward(self, input):
        """ Compute keypoints, scores, descriptors for image """
        data = input['image']
        H, W = data.shape[2:]

        if self.config['input_type'] == 'rgb':
            # 3-channel rgb
            RGB_mean = [0.485, 0.456, 0.406]
            RGB_std  = [0.229, 0.224, 0.225]
            norm_RGB = tvf.Normalize(mean=RGB_mean, std=RGB_std)
            data = norm_RGB(data)

        elif self.config['input_type'] == 'gray':
            # 1-channel
            data = torch.mean(data, dim=1, keepdim=True)
            norm_gray0 = tvf.Normalize(mean=data.mean(), std=data.std())
            data = norm_gray0(data)

        elif self.config['input_type'] == 'raw':
            # 4-channel
            pass
        elif self.config['input_type'] == 'raw-demosaic':
            # 3-channel
            pass
        else:
            raise NotImplementedError()
        
        # x: [N, C, H, W]
        x0 = self.conv0(data)
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

        comb_weights = torch.tensor([1., 2., 3.], device=data.device)
        comb_weights /= torch.sum(comb_weights)
        ksize = [3, 2, 1]
        det_score_maps = []

        for idx, xx in enumerate([x1, x3, x6_2]):
            alpha, beta = peakiness_score(xx, self.moving_avg_params[idx].detach(), ksize=3, dilation=ksize[idx])
            score_vol = alpha * beta
            det_score_map = torch.max(score_vol, dim=1, keepdim=True)[0]
            det_score_map = F.interpolate(det_score_map, size=data.shape[2:], mode='bilinear', align_corners=True)
            det_score_map = comb_weights[idx] * det_score_map
            det_score_maps.append(det_score_map)

        det_score_map = torch.sum(torch.stack(det_score_maps, dim=0), dim=0)

        desc = x6_2
        score_map = det_score_map
        conf = F.softmax(self.clf((desc)**2), dim=1)[:,1:2]
        score_map = score_map * F.interpolate(conf, size=score_map.shape[2:], mode='bilinear', align_corners=True)

        kpt_inds, kpt_score = extract_kpts(
            score_map,
            k=self.config['kpt_n'],
            score_thld=self.config['score_thld'],
            nms_size=self.config['nms_size'],
            eof_size=self.config['eof_size'],
            edge_thld=self.config['edge_thld']
        )

        descs = F.normalize(
                    interpolate(kpt_inds.squeeze(0) / 4, desc.squeeze(0).permute(1, 2, 0)),
                    p=2,
                    dim=-1
                ).detach().cpu().numpy(),
        kpts = np.squeeze(torch.stack([kpt_inds[:, :, 1], kpt_inds[:, :, 0]], dim=-1).cpu(), axis=0) \
                * np.array([W / data.shape[3], H / data.shape[2]], dtype=np.float32)
        scores = np.squeeze(kpt_score.detach().cpu().numpy(), axis=0)

        idxs = np.negative(scores).argsort()[0:self.config['kpt_n']]
        descs = descs[0][idxs]
        kpts = kpts[idxs]
        scores = scores[idxs]

        return {
            'keypoints': kpts,
            'scores': torch.from_numpy(scores),
            'descriptors': torch.from_numpy(descs.T),
        }
