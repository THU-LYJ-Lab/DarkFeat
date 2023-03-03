import torch
from torch import nn


class NN2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        desc1, desc2 = data['descriptors0'].cuda(), data['descriptors1'].cuda()
        kpts1, kpts2 = data['keypoints0'].cuda(), data['keypoints1'].cuda()

        # torch.cuda.synchronize()
        # t = time.time()

        if kpts1.shape[1] <= 1 or kpts2.shape[1] <= 1:  # no keypoints
            shape0, shape1 = kpts1.shape[:-1], kpts2.shape[:-1]
            return {
                'matches0': kpts1.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts2.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts1.new_zeros(shape0),
                'matching_scores1': kpts2.new_zeros(shape1),
            }

        sim = torch.matmul(desc1.squeeze().T, desc2.squeeze())
        ids1 = torch.arange(0, sim.shape[0], device=desc1.device)
        nn12 = torch.argmax(sim, dim=1)

        nn21 = torch.argmax(sim, dim=0)
        mask = torch.eq(ids1, nn21[nn12])
        matches = torch.stack([torch.masked_select(ids1, mask), torch.masked_select(nn12, mask)])
        # matches = torch.stack([ids1, nn12])
        indices0 = torch.ones((1, desc1.shape[-1]), dtype=int) * -1
        mscores0 = torch.ones((1, desc1.shape[-1]), dtype=float) * -1

        # torch.cuda.synchronize()
        # print(time.time() - t)
            
        matches_0 = matches[0].cpu().int().numpy()
        matches_1 = matches[1].cpu().int()
        for i in range(matches.shape[-1]):
            indices0[0, matches_0[i]] = matches_1[i].int()
            mscores0[0, matches_0[i]] = sim[matches_0[i], matches_1[i]]

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices0, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores0,
        }
