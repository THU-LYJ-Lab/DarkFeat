import argparse
import glob
import math
import subprocess
import numpy as np
import os
import tqdm
import torch
import torch.nn as nn
import cv2
from darkfeat import DarkFeat
from utils import matching

def darkfeat_pre(img, cuda):
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = inp.transpose(2, 0, 1)
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 3, H, W)
    if cuda:
        inp = inp.cuda()
    return inp

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int, default=int(640))
    parser.add_argument('--W', type=int, default=int(960))
    parser.add_argument('--histeq', action='store_true')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--dataset_dir', type=str, default='/data/hyz/MID/')
    opt = parser.parse_args()

    sizer = (opt.W, opt.H)
    focallength_x = 4.504986436499113e+03/(6744/sizer[0])
    focallength_y = 4.513311442889859e+03/(4502/sizer[1])
    K = np.eye(3)
    K[0,0] = focallength_x
    K[1,1] = focallength_y
    K[0,2] = 3.363322177533149e+03/(6744/sizer[0])# * 0.5
    K[1,2] = 2.291824660547715e+03/(4502/sizer[1])# * 0.5
    Kinv = np.linalg.inv(K)
    Kinvt = np.transpose(Kinv)

    cuda = True
    if cuda:
        darkfeat = DarkFeat(opt.model_path).cuda().eval()

    for scene in ['Indoor', 'Outdoor']:
        base_save = './result/' + scene + '/'
        dir_base = opt.dataset_dir + '/' + scene + '/'
        pair_list = sorted(os.listdir(dir_base))

        for pair in tqdm.tqdm(pair_list):
            opention = 1
            if scene == 'Outdoor':
                pass
            else:
                if int(pair[4::]) <= 17:
                    opention = 0
                else:
                    pass
            name=[]
            files = sorted(os.listdir(dir_base+pair))
            for file_ in files:
                if file_.endswith('.cr2'):
                    name.append(file_[0:9])
            ISO = ['00100', '00200', '00400', '00800', '01600', '03200', '06400', '12800']
            if opention == 1:
                Shutter_speed = ['0.005','0.01','0.025','0.05','0.17','0.5']
            else:
                Shutter_speed = ['0.01','0.02','0.05','0.1','0.3','1']

            E_GT = np.load(dir_base+pair+'/GT_Correspondence/'+'E_estimated.npy')
            F_GT = np.dot(np.dot(Kinvt,E_GT),Kinv)
            R_GT = np.load(dir_base+pair+'/GT_Correspondence/'+'R_GT.npy')
            t_GT = np.load(dir_base+pair+'/GT_Correspondence/'+'T_GT.npy')

            id0, id1 = sorted([ int(i.split('/')[-1]) for i in glob.glob(f'{dir_base+pair}/?????') ])

            cnt = 0

            for iso in ISO:
                for ex in Shutter_speed:
                    dark_name1 = name[0] + iso+'_'+ex+'_'+scene+'.npy'
                    dark_name2 = name[1] + iso+'_'+ex+'_'+scene+'.npy'

                    if not opt.histeq:
                        dst_T1_None = f'{dir_base}{pair}/{id0:05d}-npy-nohisteq/{dark_name1}'
                        dst_T2_None = f'{dir_base}{pair}/{id1:05d}-npy-nohisteq/{dark_name2}'

                        img1_orig_None = np.load(dst_T1_None)
                        img2_orig_None = np.load(dst_T2_None)

                        dir_save = base_save + pair + '/None/'

                        img_input1 = darkfeat_pre(img1_orig_None.astype('float32')/255.0, cuda)
                        img_input2 = darkfeat_pre(img2_orig_None.astype('float32')/255.0, cuda)

                    else:
                        dst_T1_histeq = f'{dir_base}{pair}/{id0:05d}-npy/{dark_name1}'
                        dst_T2_histeq = f'{dir_base}{pair}/{id1:05d}-npy/{dark_name2}'

                        img1_orig_histeq = np.load(dst_T1_histeq)
                        img2_orig_histeq = np.load(dst_T2_histeq)

                        dir_save = base_save + pair + '/HistEQ/'

                        img_input1 = darkfeat_pre(img1_orig_histeq.astype('float32')/255.0, cuda)
                        img_input2 = darkfeat_pre(img2_orig_histeq.astype('float32')/255.0, cuda)

                    result1 = darkfeat({'image': img_input1})
                    result2 = darkfeat({'image': img_input2})

                    mkpts0, mkpts1, _ = matching.match_descriptors(
                        cv2.KeyPoint_convert(result1['keypoints'].detach().cpu().float().numpy()), result1['descriptors'].detach().cpu().numpy(),
                        cv2.KeyPoint_convert(result2['keypoints'].detach().cpu().float().numpy()), result2['descriptors'].detach().cpu().numpy(),
                        ORB=False
                    )

                    POINT_1_dir = dir_save+f'DarkFeat/POINT_1/'
                    POINT_2_dir = dir_save+f'DarkFeat/POINT_2/'

                    subprocess.check_output(['mkdir', '-p', POINT_1_dir])
                    subprocess.check_output(['mkdir', '-p', POINT_2_dir])
                    np.save(POINT_1_dir+dark_name1[0:-3]+'npy',mkpts0)
                    np.save(POINT_2_dir+dark_name2[0:-3]+'npy',mkpts1)

