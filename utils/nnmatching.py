import torch

from .nn import NN2
from darkfeat import DarkFeat

class NNMatching(torch.nn.Module):
    def __init__(self, model_path=''):
        super().__init__()
        self.nn = NN2().eval()
        self.darkfeat = DarkFeat(model_path).eval()

    def forward(self, data):
        """ Run DarkFeat and nearest neighborhood matching
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract DarkFeat (keypoints, scores, descriptors)
        if 'keypoints0' not in data:
            pred0 = self.darkfeat({'image': data['image0']})
            # print({k+'0': v[0].shape for k, v in pred0.items()})
            pred = {**pred, **{k+'0': [v] for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.darkfeat({'image': data['image1']})
            pred = {**pred, **{k+'1': [v] for k, v in pred1.items()}}
        

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.nn(data)}

        return pred
