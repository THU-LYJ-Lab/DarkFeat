from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
from utils.nnmatching import NNMatching
from utils.misc import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


def compute_essential(matched_kp1, matched_kp2, K):
    pts1 = cv2.undistortPoints(matched_kp1,cameraMatrix=K, distCoeffs = (-0.117918271740560,0.075246403574314,0,0))
    pts2 = cv2.undistortPoints(matched_kp2,cameraMatrix=K, distCoeffs = (-0.117918271740560,0.075246403574314,0,0))
    K_1 = np.eye(3)
    # Estimate the homography between the matches using RANSAC
    ransac_model, ransac_inliers = cv2.findEssentialMat(pts1, pts2, K_1, method=cv2.RANSAC, prob=0.999, threshold=0.001, maxIters=10000)
    if ransac_inliers is None or ransac_model.shape != (3,3):
        ransac_inliers = np.array([])
        ransac_model = None
    return ransac_model, ransac_inliers, pts1, pts2


sizer = (960, 640)
focallength_x = 4.504986436499113e+03/(6744/sizer[0])
focallength_y = 4.513311442889859e+03/(4502/sizer[1])
K = np.eye(3)
K[0,0] = focallength_x
K[1,1] = focallength_y
K[0,2] = 3.363322177533149e+03/(6744/sizer[0])# * 0.5
K[1,2] = 2.291824660547715e+03/(4502/sizer[1])# * 0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DarkFeat demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str,
        help='path to an image directory')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.ARW'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument('--model_path', type=str,
                        help='Path to the pretrained model')

    opt = parser.parse_args()
    print(opt)

    assert len(opt.resize) == 2
    print('Will resize to {}x{} (WxH)'.format(opt.resize[0], opt.resize[1]))

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    matching = NNMatching(opt.model_path).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    vs = VideoStreamer(opt.input, opt.resize, opt.image_glob)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.darkfeat({'image': frame_tensor})
    last_data = {k+'0': [last_data[k]] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    timer = AverageTimer()

    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_darkfeat.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        E, inliers, pts1, pts2 = compute_essential(mkpts0, mkpts1, K)
        color = cm.jet(np.clip(confidence[valid][inliers[:, 0].astype('bool')] * 2 - 1, -1, 1))

        text = [
            'DarkFeat',
            'Matches: {}'.format(inliers.sum())
        ]

        out = make_matching_plot_fast(
            last_frame, frame, mkpts0[inliers[:, 0].astype('bool')], mkpts1[inliers[:, 0].astype('bool')], color, text,
            path=None, small_text=' ')

        if opt.output_dir is not None:
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('Writing image to {}'.format(out_file))
            cv2.imwrite(out_file, out)
