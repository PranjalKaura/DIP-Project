#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 23:10:58 2019

@author: prateek_prabhu
"""

import cv2
import numpy as np
in_img = cv2.imread('b.png')
cv2.imshow("in_img", in_img)


def brighten(img, val):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
#    print(v)
    lim = 255 - val

    v[v > lim] = 255
    v[v <= lim] += val

    final_hsv = cv2.merge((h,s,v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def unsharpmask(inp_img):  # unsharp masking - Image sharpening
    kernel = np.ones((5, 5), np.float32) / 25
    conv_img = cv2.filter2D(inp_img, -1, kernel)

    mask = cv2.addWeighted(inp_img, 1, conv_img, -1, 0)

    out_img = cv2.addWeighted(inp_img, 1, mask, 1, 0)
    out_img = out_img.astype(np.uint8)
    out_img = out_img.astype(np.uint8)
    return out_img;


def darken(image, gamma):
    invG = 1.0/gamma
    table = np.array([((i / 255.0) ** invG) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


out1 = brighten(in_img, 50)
out2 = darken(in_img, 0.5)
out3 = unsharpmask(out1)
#cv2.imshow("out_img1", out1)
cv2.imshow("out_img2", out2)
cv2.imshow("mask", out3)
cv2.waitKey()
