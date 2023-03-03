import torch
import torch.nn.functional as F

from .geom import rnd_sample, interpolate, get_dist_mat


def make_detector_loss(pos0, pos1, dense_feat_map0, dense_feat_map1,
               score_map0, score_map1, batch_size, num_corr, loss_type, config):
    joint_loss = 0.
    accuracy = 0.
    all_valid_pos0 = []
    all_valid_pos1 = []
    all_valid_match = []
    for i in range(batch_size):
        # random sample
        valid_pos0, valid_pos1 = rnd_sample([pos0[i], pos1[i]], num_corr)
        valid_num = valid_pos0.shape[0]

        valid_feat0 = interpolate(valid_pos0 / 4, dense_feat_map0[i])
        valid_feat1 = interpolate(valid_pos1 / 4, dense_feat_map1[i])

        valid_feat0 = F.normalize(valid_feat0, p=2, dim=-1)
        valid_feat1 = F.normalize(valid_feat1, p=2, dim=-1)

        valid_score0 = interpolate(valid_pos0, torch.squeeze(score_map0[i], dim=-1), nd=False)
        valid_score1 = interpolate(valid_pos1, torch.squeeze(score_map1[i], dim=-1), nd=False)
            
        if config['network']['det']['corr_weight']:
            corr_weight = valid_score0 * valid_score1
        else:
            corr_weight = None

        safe_radius = config['network']['det']['safe_radius']
        if safe_radius > 0:
            radius_mask_row = get_dist_mat(
                valid_pos1, valid_pos1, "euclidean_dist_no_norm")
            radius_mask_row = torch.le(radius_mask_row, safe_radius)
            radius_mask_col = get_dist_mat(
                valid_pos0, valid_pos0, "euclidean_dist_no_norm")
            radius_mask_col = torch.le(radius_mask_col, safe_radius)
            radius_mask_row = radius_mask_row.float() - torch.eye(valid_num, device=radius_mask_row.device)
            radius_mask_col = radius_mask_col.float() - torch.eye(valid_num, device=radius_mask_col.device)
        else:
            radius_mask_row = None
            radius_mask_col = None

        if valid_num < 32:
            si_loss, si_accuracy, matched_mask = 0., 1., torch.zeros((1, valid_num)).bool()
        else:
            si_loss, si_accuracy, matched_mask = make_structured_loss(
                torch.unsqueeze(valid_feat0, 0), torch.unsqueeze(valid_feat1, 0),
                loss_type=loss_type,
                radius_mask_row=radius_mask_row, radius_mask_col=radius_mask_col,
                corr_weight=torch.unsqueeze(corr_weight, 0) if corr_weight is not None else None
            )

        joint_loss += si_loss / batch_size
        accuracy += si_accuracy / batch_size
        all_valid_match.append(torch.squeeze(matched_mask, dim=0))
        all_valid_pos0.append(valid_pos0)
        all_valid_pos1.append(valid_pos1)

    return joint_loss, accuracy


def make_structured_loss(feat_anc, feat_pos,
                         loss_type='RATIO', inlier_mask=None,
                         radius_mask_row=None, radius_mask_col=None,
                         corr_weight=None, dist_mat=None):
    """
    Structured loss construction.
    Args:
        feat_anc, feat_pos: Feature matrix.
        loss_type: Loss type.
        inlier_mask:
    Returns:

    """
    batch_size = feat_anc.shape[0]
    num_corr = feat_anc.shape[1]
    if inlier_mask is None:
        inlier_mask = torch.ones((batch_size, num_corr), device=feat_anc.device).bool()
    inlier_num = torch.count_nonzero(inlier_mask.float(), dim=-1)

    if loss_type == 'L2NET' or loss_type == 'CIRCLE':
        dist_type = 'cosine_dist'
    elif loss_type.find('HARD') >= 0:
        dist_type = 'euclidean_dist'
    else:
        raise NotImplementedError()

    if dist_mat is None:
        dist_mat = get_dist_mat(feat_anc.squeeze(0), feat_pos.squeeze(0), dist_type).unsqueeze(0)
    pos_vec = dist_mat[0].diag().unsqueeze(0)

    if loss_type.find('HARD') >= 0:
        neg_margin = 1
        dist_mat_without_min_on_diag = dist_mat + \
            10 * torch.unsqueeze(torch.eye(num_corr, device=dist_mat.device), dim=0)
        mask = torch.le(dist_mat_without_min_on_diag, 0.008).float()
        dist_mat_without_min_on_diag += mask*10

        if radius_mask_row is not None:
            hard_neg_dist_row = dist_mat_without_min_on_diag + 10 * radius_mask_row
        else:
            hard_neg_dist_row = dist_mat_without_min_on_diag
        if radius_mask_col is not None:
            hard_neg_dist_col = dist_mat_without_min_on_diag + 10 * radius_mask_col
        else:
            hard_neg_dist_col = dist_mat_without_min_on_diag

        hard_neg_dist_row = torch.min(hard_neg_dist_row, dim=-1)[0]
        hard_neg_dist_col = torch.min(hard_neg_dist_col, dim=-2)[0]

        if loss_type == 'HARD_TRIPLET':
            loss_row = torch.clamp(neg_margin + pos_vec - hard_neg_dist_row, min=0)
            loss_col = torch.clamp(neg_margin + pos_vec - hard_neg_dist_col, min=0)
        elif loss_type == 'HARD_CONTRASTIVE':
            pos_margin = 0.2
            pos_loss = torch.clamp(pos_vec - pos_margin, min=0)
            loss_row = pos_loss + torch.clamp(neg_margin - hard_neg_dist_row, min=0)
            loss_col = pos_loss + torch.clamp(neg_margin - hard_neg_dist_col, min=0)
        else:
            raise NotImplementedError()
    
    elif loss_type == 'CIRCLE':
        log_scale = 512
        m = 0.1
        neg_mask_row = torch.unsqueeze(torch.eye(num_corr, device=feat_anc.device), 0)
        if radius_mask_row is not None:
            neg_mask_row += radius_mask_row
        neg_mask_col = torch.unsqueeze(torch.eye(num_corr, device=feat_anc.device), 0)
        if radius_mask_col is not None:
            neg_mask_col += radius_mask_col

        pos_margin = 1 - m
        neg_margin = m
        pos_optimal = 1 + m
        neg_optimal = -m

        neg_mat_row = dist_mat - 128 * neg_mask_row
        neg_mat_col = dist_mat - 128 * neg_mask_col

        lse_positive = torch.logsumexp(-log_scale * (pos_vec[..., None] - pos_margin) * \
                    torch.clamp(pos_optimal - pos_vec[..., None], min=0).detach(), dim=-1)
        
        lse_negative_row = torch.logsumexp(log_scale * (neg_mat_row - neg_margin) * \
                    torch.clamp(neg_mat_row - neg_optimal, min=0).detach(), dim=-1)

        lse_negative_col = torch.logsumexp(log_scale * (neg_mat_col - neg_margin) * \
                    torch.clamp(neg_mat_col - neg_optimal, min=0).detach(), dim=-2)

        loss_row = F.softplus(lse_positive + lse_negative_row) / log_scale
        loss_col = F.softplus(lse_positive + lse_negative_col) / log_scale

    else:
        raise NotImplementedError()

    if dist_type == 'cosine_dist':
        err_row = dist_mat - torch.unsqueeze(pos_vec, -1)
        err_col = dist_mat - torch.unsqueeze(pos_vec, -2)
    elif dist_type == 'euclidean_dist' or dist_type == 'euclidean_dist_no_norm':
        err_row = torch.unsqueeze(pos_vec, -1) - dist_mat
        err_col = torch.unsqueeze(pos_vec, -2) - dist_mat
    else:
        raise NotImplementedError()
    if radius_mask_row is not None:
        err_row = err_row - 10 * radius_mask_row
    if radius_mask_col is not None:
        err_col = err_col - 10 * radius_mask_col
    err_row = torch.sum(torch.clamp(err_row, min=0), dim=-1)
    err_col = torch.sum(torch.clamp(err_col, min=0), dim=-2)

    loss = 0
    accuracy = 0

    tot_loss = (loss_row + loss_col) / 2
    if corr_weight is not None:
        tot_loss = tot_loss * corr_weight

    for i in range(batch_size):
        if corr_weight is not None:
            loss += torch.sum(tot_loss[i][inlier_mask[i]]) / \
                (torch.sum(corr_weight[i][inlier_mask[i]]) + 1e-6)
        else:
            loss += torch.mean(tot_loss[i][inlier_mask[i]])
        cnt_err_row = torch.count_nonzero(err_row[i][inlier_mask[i]]).float()
        cnt_err_col = torch.count_nonzero(err_col[i][inlier_mask[i]]).float()
        tot_err = cnt_err_row + cnt_err_col
        if inlier_num[i] != 0:
            accuracy += 1. - tot_err / inlier_num[i] / batch_size / 2.
        else:
            accuracy += 1.

    matched_mask = torch.logical_and(torch.eq(err_row, 0), torch.eq(err_col, 0))
    matched_mask = torch.logical_and(matched_mask, inlier_mask)

    loss /= batch_size
    accuracy /= batch_size

    return loss, accuracy, matched_mask


# for the neighborhood areas of keypoints extracted from normal image, the score from noise_score_map should be close
# for the rest, the noise image's score should less than normal image
# input: score_map [batch_size, H, W, 1]; indices [2, k, 2]
# output: loss [scalar]
def make_noise_score_map_loss(score_map, noise_score_map, indices, batch_size, thld=0.):
    H, W = score_map.shape[1:3]
    loss = 0
    for i in range(batch_size):
        kpts_coords = indices[i].T # (2, num_kpts)
        mask = torch.zeros([H, W], device=score_map.device)
        mask[kpts_coords.cpu().numpy()] = 1

        # using 3x3 kernel to put kpts' neightborhood area into the mask
        kernel = torch.ones([1, 1, 3, 3], device=score_map.device)
        mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)[0, 0] > 0

        loss1 = torch.sum(torch.abs(score_map[i] - noise_score_map[i]).squeeze() * mask) / torch.sum(mask)
        loss2 = torch.sum(torch.clamp(noise_score_map[i] - score_map[i] - thld, min=0).squeeze() * torch.logical_not(mask)) / (H * W - torch.sum(mask))

        loss += loss1
        loss += loss2

        if i == 0:
            first_mask = mask

    return loss, first_mask


def make_noise_score_map_loss_labelmap(score_map, noise_score_map, labelmap, batch_size, thld=0.):
    H, W = score_map.shape[1:3]
    loss = 0
    for i in range(batch_size):
        # using 3x3 kernel to put kpts' neightborhood area into the mask
        kernel = torch.ones([1, 1, 3, 3], device=score_map.device)
        mask = F.conv2d(labelmap[i].unsqueeze(0).to(score_map.device).float(), kernel, padding=1)[0, 0] > 0

        loss1 = torch.sum(torch.abs(score_map[i] - noise_score_map[i]).squeeze() * mask) / torch.sum(mask)
        loss2 = torch.sum(torch.clamp(noise_score_map[i] - score_map[i] - thld, min=0).squeeze() * torch.logical_not(mask)) / (H * W - torch.sum(mask))

        loss += loss1
        loss += loss2

        if i == 0:
            first_mask = mask

    return loss, first_mask


def make_score_map_peakiness_loss(score_map, scores, batch_size):
    H, W = score_map.shape[1:3]
    loss = 0

    for i in range(batch_size):
        loss += torch.mean(scores[i]) - torch.mean(score_map[i])

    loss /= batch_size
    return 1 - loss
