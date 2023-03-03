import os
import numpy as np
import torch
from torch.utils.data import Dataset
from random import shuffle, seed

from .gl3d.io import read_list, _parse_img, _parse_depth, _parse_kpts
from .utils.common import Notify
from .utils.photaug import photaug


class GL3DDataset(Dataset):
    def __init__(self, dataset_dir, config, data_split, is_training):
        self.dataset_dir = dataset_dir
        self.config = config
        self.is_training = is_training
        self.data_split = data_split
        
        self.match_set_list, self.global_img_list, \
            self.global_depth_list = self.prepare_match_sets()

        pass


    def __len__(self):
        return len(self.match_set_list)


    def __getitem__(self, idx):
        match_set_path = self.match_set_list[idx]
        decoded = np.fromfile(match_set_path, dtype=np.float32)

        idx0, idx1 = int(decoded[0]), int(decoded[1])
        inlier_num = int(decoded[2])
        ori_img_size0 = np.reshape(decoded[3:5], (2,))
        ori_img_size1 = np.reshape(decoded[5:7], (2,))
        K0 = np.reshape(decoded[7:16], (3, 3))
        K1 = np.reshape(decoded[16:25], (3, 3))
        rel_pose = np.reshape(decoded[34:46], (3, 4))

        # parse images.
        img0 = _parse_img(self.global_img_list, idx0, self.config)
        img1 = _parse_img(self.global_img_list, idx1, self.config)
        # parse depths
        depth0 = _parse_depth(self.global_depth_list, idx0, self.config)
        depth1 = _parse_depth(self.global_depth_list, idx1, self.config)

        # photometric augmentation
        img0 = photaug(img0)
        img1 = photaug(img1)

        return {
            'img0': img0 / 255.,
            'img1': img1 / 255.,
            'depth0': depth0,
            'depth1': depth1,
            'ori_img_size0': ori_img_size0,
            'ori_img_size1': ori_img_size1,
            'K0': K0,
            'K1': K1,
            'rel_pose': rel_pose,
            'inlier_num': inlier_num
        }


    def points_to_2D(self, pnts, H, W):
        labels = np.zeros((H, W))
        pnts = pnts.astype(int)
        labels[pnts[:, 1], pnts[:, 0]] = 1
        return labels


    def prepare_match_sets(self, q_diff_thld=3, rot_diff_thld=60):
        """Get match sets.
        Args:
            is_training: Use training imageset or testing imageset.
            data_split: Data split name.
        Returns:
            match_set_list: List of match sets path.
            global_img_list: List of global image path.
            global_context_feat_list:
        """
        # get necessary lists.
        gl3d_list_folder = os.path.join(self.dataset_dir, 'list', self.data_split)
        global_info = read_list(os.path.join(
            gl3d_list_folder, 'image_index_offset.txt'))
        global_img_list = [os.path.join(self.dataset_dir, i) for i in read_list(
            os.path.join(gl3d_list_folder, 'image_list.txt'))]
        global_depth_list = [os.path.join(self.dataset_dir, i) for i in read_list(
            os.path.join(gl3d_list_folder, 'depth_list.txt'))]

        imageset_list_name = 'imageset_train.txt' if self.is_training else 'imageset_test.txt'
        match_set_list = self.get_match_set_list(os.path.join(
            gl3d_list_folder, imageset_list_name), q_diff_thld, rot_diff_thld)
        return match_set_list, global_img_list, global_depth_list


    def get_match_set_list(self, imageset_list_path, q_diff_thld, rot_diff_thld):
        """Get the path list of match sets.
        Args:
            imageset_list_path: Path to imageset list.
            q_diff_thld: Threshold of image pair sampling regarding camera orientation.
        Returns:
            match_set_list: List of match set path.
        """
        imageset_list = [os.path.join(self.dataset_dir, 'data', i)
                        for i in read_list(imageset_list_path)]
        print(Notify.INFO, 'Use # imageset', len(imageset_list), Notify.ENDC)
        match_set_list = []
        # discard image pairs whose image simiarity is beyond the threshold.
        for i in imageset_list:
            match_set_folder = os.path.join(i, 'match_sets')
            if os.path.exists(match_set_folder):
                match_set_files = os.listdir(match_set_folder)
                for val in match_set_files:
                    name, ext = os.path.splitext(val)
                    if ext == '.match_set':
                        splits = name.split('_')
                        q_diff = int(splits[2])
                        rot_diff = int(splits[3])
                        if q_diff >= q_diff_thld and rot_diff <= rot_diff_thld:
                            match_set_list.append(
                                os.path.join(match_set_folder, val))

        print(Notify.INFO, 'Get # match sets', len(match_set_list), Notify.ENDC)
        return match_set_list
        
