import math
import numpy as np
import cv2

def extract_ORB_keypoints_and_descriptors(img):
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=1000)
    kp, desc = detector.detectAndCompute(img, None)
    return kp, desc

def match_descriptors_NG(kp1, desc1, kp2, desc2):
    bf = cv2.BFMatcher()
    try:
        matches = bf.knnMatch(desc1, desc2,k=2)
    except:
        matches = []
    good_matches=[]
    image1_kp = []
    image2_kp = []
    ratios = []
    try:
        for (m1,m2) in matches:
            if m1.distance < 0.8 * m2.distance:
                good_matches.append(m1)
                image2_kp.append(kp2[m1.trainIdx].pt)
                image1_kp.append(kp1[m1.queryIdx].pt)
                ratios.append(m1.distance / m2.distance)
    except:
        pass
    image1_kp = np.array([image1_kp])
    image2_kp = np.array([image2_kp])
    ratios = np.array([ratios])
    ratios = np.expand_dims(ratios, 2)
    return image1_kp, image2_kp, good_matches, ratios

def match_descriptors(kp1, desc1, kp2, desc2, ORB):
    if ORB:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(desc1,desc2)
            matches = sorted(matches, key = lambda x:x.distance)
        except:
            matches = []
        good_matches=[]
        image1_kp = []
        image2_kp = []
        count = 0
        try:
            for m in matches:
                count+=1
                if count < 1000:
                    good_matches.append(m)
                    image2_kp.append(kp2[m.trainIdx].pt)
                    image1_kp.append(kp1[m.queryIdx].pt)  
        except:
            pass
    else:
        # Match the keypoints with the warped_keypoints with nearest neighbor search
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        try:
            matches = bf.match(desc1.transpose(1,0), desc2.transpose(1,0))  
            matches = sorted(matches, key = lambda x:x.distance)
        except:
            matches = []
        good_matches=[]
        image1_kp = []
        image2_kp = []
        try:
            for m in matches:
                good_matches.append(m)              
                image2_kp.append(kp2[m.trainIdx].pt)
                image1_kp.append(kp1[m.queryIdx].pt)
        except:
            pass

    image1_kp = np.array([image1_kp])
    image2_kp = np.array([image2_kp])
    return image1_kp, image2_kp, good_matches


def compute_essential(matched_kp1, matched_kp2, K):
    pts1 = cv2.undistortPoints(matched_kp1,cameraMatrix=K, distCoeffs = (-0.117918271740560,0.075246403574314,0,0))
    pts2 = cv2.undistortPoints(matched_kp2,cameraMatrix=K, distCoeffs = (-0.117918271740560,0.075246403574314,0,0))
    K_1 = np.eye(3)
    # Estimate the homography between the matches using RANSAC
    ransac_model, ransac_inliers = cv2.findEssentialMat(pts1, pts2, K_1, method=cv2.FM_RANSAC, prob=0.999, threshold=0.001)
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
        cv2.recoverPose(E, pts1_norm, pts2_norm, np.eye(3), R, t, inliers)
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
        dR,dT = 180.0, 180.0
    return dR, dT
