import numpy as np
import random
from scipy.stats import tukeylambda

camera_params = {
    'Kmin': 0.2181895124454343,
    'Kmax': 3.0,
    'G_shape': np.array([0.15714286, 0.14285714, 0.08571429, 0.08571429, 0.2       ,
                         0.2       , 0.1       , 0.08571429, 0.05714286, 0.07142857,
                         0.02857143, 0.02857143, 0.01428571, 0.02857143, 0.08571429,
                         0.07142857, 0.11428571, 0.11428571]),
    'Profile-1': {
        'R_scale': {
            'slope': 0.4712797750747537,
            'bias': -0.8078958947116487,
            'sigma': 0.2436176299944695
        },
        'g_scale': {
            'slope': 0.6771267783987617,
            'bias': 1.5121876510805845,
            'sigma': 0.24641096601611254
        },
        'G_scale': {
            'slope': 0.6558756156508007,
            'bias': 1.09268679594838,
            'sigma': 0.28604721742277756
        }
    },
    'black_level': 2048,
    'max_value': 16383
}


# photon shot noise
def addPStarNoise(img, K):
    return np.random.poisson(img / K).astype(np.float32) * K


# read noise
# tukey lambda distribution
def addGStarNoise(img, K, G_shape, G_scale_param):
    # sample a shape parameter [lambda] from histogram of samples
    a, b = np.histogram(G_shape, bins=10, range=(-0.25, 0.25))
    a, b = np.array(a), np.array(b)
    a = a / a.sum()

    rand_num = random.uniform(0, 1)
    idx = np.sum(np.cumsum(a) < rand_num)
    lam = random.uniform(b[idx], b[idx+1])

    # calculate scale parameter [G_scale]
    log_K = np.log(K)
    log_G_scale = np.random.standard_normal() * G_scale_param['sigma'] * 1 +\
             G_scale_param['slope'] * log_K + G_scale_param['bias']
    G_scale = np.exp(log_G_scale)
    # print(f'G_scale: {G_scale}')
    
    return img + tukeylambda.rvs(lam, scale=G_scale, size=img.shape).astype(np.float32)


# row noise
# uniform distribution for each row
def addRowNoise(img, K, R_scale_param):
    # calculate scale parameter [R_scale]
    log_K = np.log(K)
    log_R_scale = np.random.standard_normal() * R_scale_param['sigma'] * 1 +\
             R_scale_param['slope'] * log_K + R_scale_param['bias']
    R_scale = np.exp(log_R_scale)
    # print(f'R_scale: {R_scale}')
    
    row_noise = np.random.randn(img.shape[0], 1).astype(np.float32) * R_scale
    return img + np.tile(row_noise, (1, img.shape[1]))


# quantization noise
# uniform distribution
def addQuantNoise(img, q):
    return img + np.random.uniform(low=-0.5*q, high=0.5*q, size=img.shape)


def sampleK(Kmin, Kmax):
    return np.exp(np.random.uniform(low=np.log(Kmin), high=np.log(Kmax)))
