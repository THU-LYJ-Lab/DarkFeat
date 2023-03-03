import argparse
import cv2
import numpy as np
import os
import math
import subprocess
from tqdm import tqdm


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


def compute_error(R_GT,t_GT,E,pts1_norm, pts2_norm, inliers):
    """Compute the angular error between two rotation matrices and two translation vectors.
    Keyword arguments:
    R -- 2D numpy array containing an estimated rotation
    gt_R -- 2D numpy array containing the corresponding ground truth rotation
    t -- 2D numpy array containing an estimated translation as column
    gt_t -- 2D numpy array containing the corresponding ground truth translation
    """

    inliers = inliers.ravel()
    R = np.eye(3)
    t = np.zeros((3,1))
    sst = True
    try:
        _, R, t, _ = cv2.recoverPose(E, pts1_norm, pts2_norm, np.eye(3), inliers)
    except:
        sst = False
    # calculate angle between provided rotations
    # 
    if sst:
        dR = np.matmul(R, np.transpose(R_GT))
        dR = cv2.Rodrigues(dR)[0]
        dR = np.linalg.norm(dR) * 180 / math.pi

        # calculate angle between provided translations
        dT = float(np.dot(t_GT.T, t))
        dT /= float(np.linalg.norm(t_GT))

        if dT > 1 or dT < -1:
            print("Domain warning! dT:",dT)
            dT = max(-1,min(1,dT))
        dT = math.acos(dT) * 180 / math.pi
        dT = np.minimum(dT, 180 - dT) # ambiguity of E estimation
    else:
        dR, dT = 180.0, 180.0
    return dR, dT


def pose_evaluation(result_base_dir, dark_name1, dark_name2, enhancer, K, R_GT, t_GT):
    try:
        m_kp1 = np.load(result_base_dir+enhancer+'/DarkFeat/POINT_1/'+dark_name1)
        m_kp2 = np.load(result_base_dir+enhancer+'/DarkFeat/POINT_2/'+dark_name2)
    except:
        return 180.0, 180.0
    try:
        E, inliers, pts1, pts2 = compute_essential(m_kp1, m_kp2, K)
    except:
        E, inliers, pts1, pts2 = np.zeros((3, 3)), np.array([]), None, None
    dR, dT = compute_error(R_GT, t_GT, E, pts1, pts2, inliers)
    return dR, dT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--histeq', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='/data/hyz/MID/')
    opt = parser.parse_args()
    
    sizer = (960, 640)
    focallength_x = 4.504986436499113e+03/(6744/sizer[0])
    focallength_y = 4.513311442889859e+03/(4502/sizer[1])
    K = np.eye(3)
    K[0,0] = focallength_x
    K[1,1] = focallength_y
    K[0,2] = 3.363322177533149e+03/(6744/sizer[0])
    K[1,2] = 2.291824660547715e+03/(4502/sizer[1])
    Kinv = np.linalg.inv(K)
    Kinvt = np.transpose(Kinv)

    PE_MT = np.zeros((6, 8))

    enhancer = 'None' if not opt.histeq else 'HistEQ'

    for scene in ['Indoor', 'Outdoor']:
        dir_base = opt.dataset_dir + '/' + scene + '/'
        base_save = 'result_errors/' + scene + '/'
        pair_list = sorted(os.listdir(dir_base))

        os.makedirs(base_save, exist_ok=True)

        for pair in tqdm(pair_list):
            opention = 1
            if scene == 'Outdoor':
                pass
            else:
                if int(pair[4::]) <= 17:
                    opention = 0
                else:
                    pass
            name = []
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
            result_base_dir ='result/' +scene+'/'+pair+'/'
            for iso in ISO:
                for ex in Shutter_speed:
                    dark_name1 = name[0]+iso+'_'+ex+'_'+scene+'.npy'
                    dark_name2 = name[1]+iso+'_'+ex+'_'+scene+'.npy'

                    dr, dt = pose_evaluation(result_base_dir,dark_name1,dark_name2,enhancer,K,R_GT,t_GT) 
                    PE_MT[Shutter_speed.index(ex),ISO.index(iso)] = max(dr, dt)

                    subprocess.check_output(['mkdir', '-p', base_save + pair + f'/{enhancer}/'])
                    np.save(base_save + pair + f'/{enhancer}/Pose_error_DarkFeat.npy', PE_MT)
          