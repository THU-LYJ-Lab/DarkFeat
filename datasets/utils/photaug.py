import cv2
import numpy as np
import random


def random_brightness_np(image, max_abs_change=50):
    delta = random.uniform(-max_abs_change, max_abs_change)
    return np.clip(image + delta, 0, 255)

def random_contrast_np(image, strength_range=[0.3, 1.5]):
    delta = random.uniform(*strength_range)
    mean = image.mean()
    return np.clip((image - mean) * delta + mean, 0, 255)

def motion_blur_np(img, max_kernel_size=3):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(
        0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center) +
                        np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img, -1, kernel)
    return np.clip(img, 0, 255)

def additive_gaussian_noise(image, stddev_range=[5, 95]):
    stddev = random.uniform(*stddev_range)
    noise = np.random.normal(size=image.shape, scale=stddev)
    noisy_image = np.clip(image + noise, 0, 255)
    return noisy_image

def photaug(img):
    img = random_brightness_np(img)
    img = random_contrast_np(img)
    # img = additive_gaussian_noise(img)
    img = motion_blur_np(img)
    return img
