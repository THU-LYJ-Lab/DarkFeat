import time
import numpy as np
import torch
import torch.nn.functional as F


def rnd_sample(inputs, n_sample):
    cur_size = inputs[0].shape[0]
    rnd_idx = torch.randperm(cur_size)[0:n_sample]
    outputs = [i[rnd_idx] for i in inputs]
    return outputs


def _grid_positions(h, w, bs):
    x_rng = torch.arange(0, w.int())
    y_rng = torch.arange(0, h.int())
    xv, yv = torch.meshgrid(x_rng, y_rng, indexing='xy')
    return torch.reshape(
        torch.stack((yv, xv), axis=-1),
        (1, -1, 2)
    ).repeat(bs, 1, 1).float()


def getK(ori_img_size, cur_feat_size, K):
    # WARNING: cur_feat_size's order is [h, w]
    r = ori_img_size / cur_feat_size[[1, 0]]
    r_K0 = torch.stack([K[:, 0] / r[:, 0][..., None], K[:, 1] /
                        r[:, 1][..., None], K[:, 2]], axis=1)
    return r_K0


def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    """
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


def validate_and_interpolate(pos, inputs, validate_corner=True, validate_val=None, nd=False):
    if nd:
        h, w, c = inputs.shape
    else:
        h, w = inputs.shape
    ids = torch.arange(0, pos.shape[0])

    i = pos[:, 0]
    j = pos[:, 1]

    i_top_left = torch.floor(i).int()
    j_top_left = torch.floor(j).int()

    i_top_right = torch.floor(i).int()
    j_top_right = torch.ceil(j).int()

    i_bottom_left = torch.ceil(i).int()
    j_bottom_left = torch.floor(j).int()

    i_bottom_right = torch.ceil(i).int()
    j_bottom_right = torch.ceil(j).int()

    if validate_corner:
        # Valid corner
        valid_top_left = torch.logical_and(i_top_left >= 0, j_top_left >= 0)
        valid_top_right = torch.logical_and(i_top_right >= 0, j_top_right < w)
        valid_bottom_left = torch.logical_and(i_bottom_left < h, j_bottom_left >= 0)
        valid_bottom_right = torch.logical_and(i_bottom_right < h, j_bottom_right < w)

        valid_corner = torch.logical_and(
            torch.logical_and(valid_top_left, valid_top_right),
            torch.logical_and(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corner]
        j_top_left = j_top_left[valid_corner]

        i_top_right = i_top_right[valid_corner]
        j_top_right = j_top_right[valid_corner]

        i_bottom_left = i_bottom_left[valid_corner]
        j_bottom_left = j_bottom_left[valid_corner]

        i_bottom_right = i_bottom_right[valid_corner]
        j_bottom_right = j_bottom_right[valid_corner]

        ids = ids[valid_corner]

    if validate_val is not None:
        # Valid depth
        valid_depth = torch.logical_and(
            torch.logical_and(
                gather_nd(inputs, torch.stack([i_top_left, j_top_left], axis=-1)) > 0,
                gather_nd(inputs, torch.stack([i_top_right, j_top_right], axis=-1)) > 0
            ),
            torch.logical_and(
                gather_nd(inputs, torch.stack([i_bottom_left, j_bottom_left], axis=-1)) > 0,
                gather_nd(inputs, torch.stack([i_bottom_right, j_bottom_right], axis=-1)) > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]

        ids = ids[valid_depth]

    # Interpolation
    i = i[ids]
    j = j[ids]
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
        w_bottom_right * gather_nd(inputs, torch.stack([i_bottom_right, j_bottom_right], axis=-1))
    )

    pos = torch.stack([i, j], axis=1)
    return [interpolated_val, pos, ids]


# pos0: [2, 230400, 2]
# depth0: [2, 480, 480]
def getWarp(pos0, rel_pose, depth0, K0, depth1, K1, bs):
    def swap_axis(data):
        return torch.stack([data[:, 1], data[:, 0]], axis=-1)

    all_pos0 = []
    all_pos1 = []
    all_ids = []
    for i in range(bs):
        z0, new_pos0, ids = validate_and_interpolate(pos0[i], depth0[i], validate_val=0)

        uv0_homo = torch.cat([swap_axis(new_pos0), torch.ones((new_pos0.shape[0], 1)).to(new_pos0.device)], axis=-1)
        xy0_homo = torch.matmul(torch.linalg.inv(K0[i]), uv0_homo.t())
        xyz0_homo = torch.cat([torch.unsqueeze(z0, 0) * xy0_homo,
                               torch.ones((1, new_pos0.shape[0])).to(z0.device)], axis=0)

        xyz1 = torch.matmul(rel_pose[i], xyz0_homo)
        xy1_homo = xyz1 / torch.unsqueeze(xyz1[-1, :], axis=0)
        uv1 = torch.matmul(K1[i], xy1_homo).t()[:, 0:2]

        new_pos1 = swap_axis(uv1)
        annotated_depth, new_pos1, new_ids = validate_and_interpolate(
            new_pos1, depth1[i], validate_val=0)

        ids = ids[new_ids]
        new_pos0 = new_pos0[new_ids]
        estimated_depth = xyz1.t()[new_ids][:, -1]

        inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05

        all_ids.append(ids[inlier_mask])
        all_pos0.append(new_pos0[inlier_mask])
        all_pos1.append(new_pos1[inlier_mask])
    # all_pos0 & all_pose1: [inlier_num, 2] * batch_size
    return all_pos0, all_pos1, all_ids


# pos0: [2, 230400, 2]
# depth0: [2, 480, 480]
def getWarpNoValidate(pos0, rel_pose, depth0, K0, depth1, K1, bs):
    def swap_axis(data):
        return torch.stack([data[:, 1], data[:, 0]], axis=-1)

    all_pos0 = []
    all_pos1 = []
    all_ids = []
    for i in range(bs):
        z0, new_pos0, ids = validate_and_interpolate(pos0[i], depth0[i], validate_val=0)

        uv0_homo = torch.cat([swap_axis(new_pos0), torch.ones((new_pos0.shape[0], 1)).to(new_pos0.device)], axis=-1)
        xy0_homo = torch.matmul(torch.linalg.inv(K0[i]), uv0_homo.t())
        xyz0_homo = torch.cat([torch.unsqueeze(z0, 0) * xy0_homo,
                               torch.ones((1, new_pos0.shape[0])).to(z0.device)], axis=0)

        xyz1 = torch.matmul(rel_pose[i], xyz0_homo)
        xy1_homo = xyz1 / torch.unsqueeze(xyz1[-1, :], axis=0)
        uv1 = torch.matmul(K1[i], xy1_homo).t()[:, 0:2]

        new_pos1 = swap_axis(uv1)
        _, new_pos1, new_ids = validate_and_interpolate(
            new_pos1, depth1[i], validate_val=0)

        ids = ids[new_ids]
        new_pos0 = new_pos0[new_ids]

        all_ids.append(ids)
        all_pos0.append(new_pos0)
        all_pos1.append(new_pos1)
    # all_pos0 & all_pose1: [inlier_num, 2] * batch_size
    return all_pos0, all_pos1, all_ids


# pos0: [2, 230400, 2]
# depth0: [2, 480, 480]
def getWarpNoValidate2(pos0, rel_pose, depth0, K0, depth1, K1):
    def swap_axis(data):
        return torch.stack([data[:, 1], data[:, 0]], axis=-1)

    z0 = interpolate(pos0, depth0, nd=False)

    uv0_homo = torch.cat([swap_axis(pos0), torch.ones((pos0.shape[0], 1)).to(pos0.device)], axis=-1)
    xy0_homo = torch.matmul(torch.linalg.inv(K0), uv0_homo.t())
    xyz0_homo = torch.cat([torch.unsqueeze(z0, 0) * xy0_homo,
                            torch.ones((1, pos0.shape[0])).to(z0.device)], axis=0)

    xyz1 = torch.matmul(rel_pose, xyz0_homo)
    xy1_homo = xyz1 / torch.unsqueeze(xyz1[-1, :], axis=0)
    uv1 = torch.matmul(K1, xy1_homo).t()[:, 0:2]

    new_pos1 = swap_axis(uv1)

    return new_pos1



def get_dist_mat(feat1, feat2, dist_type):
    eps = 1e-6
    cos_dist_mat = torch.matmul(feat1, feat2.t())
    if dist_type == 'cosine_dist':
        dist_mat = torch.clamp(cos_dist_mat, -1, 1)
    elif dist_type == 'euclidean_dist':
        dist_mat = torch.sqrt(torch.clamp(2 - 2 * cos_dist_mat, min=eps))
    elif dist_type == 'euclidean_dist_no_norm':
        norm1 = torch.sum(feat1 * feat1, axis=-1, keepdims=True)
        norm2 = torch.sum(feat2 * feat2, axis=-1, keepdims=True)
        dist_mat = torch.sqrt(
            torch.clamp(
                norm1 - 2 * cos_dist_mat + norm2.t(),
                min=0.
            ) + eps
        )
    else:
        raise NotImplementedError()
    return dist_mat
