import torch
import torch.nn.functional as F
import numpy as np

from .geom import gather_nd

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

    alpha = F.softplus(inputs - avg_spatial_inputs)
    beta = F.softplus(inputs - avg_channel_inputs)

    return alpha, beta


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
