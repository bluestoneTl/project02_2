import random

import cv2
import numpy as np
import torch



#=========新数据增强调用=========
def augment(img, hflip=True, rot=True, mode=None, swap=None):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img
    if mode in ['LQ','GT', 'SRker']:
        return _augment(img)
    elif mode in ['LQGTker', 'LQGT']:
        if swap and random.random() < 0.5:
            img.reverse()
        return [_augment(I) for I in img]
#=========新数据增强调用=========

