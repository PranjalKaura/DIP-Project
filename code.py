# -*- coding: utf-8 -*-
import numpy as np
import cv2


inp_img = cv2.imread('cameraman.tif', 0); #input image/frames

def darkGamma(inp_img): #gamma correction for dark image - light correction
    g = 1,4;
    out_img = np.power(inp_img, g);
    cv2.imshow(out_img);
    #return out_img;
    
    
def lightGamma(inp_img): #gamma correction for light image - light correction
    g = 0.4;
    out_img = np.power(inp_img, g);
    cv2.imshow(out_img);
    #return out_img;
    
def histogramEQ(inp_img): #histogram equaliation - contrast correction
    out_img = cv2.equalizeHist(inp_img);
    cv2.imshow(out_img);
    #return out_img;
    
def unsharpMask(inp_img): #unsharp masking - Image sharpening
    kernel = np.ones((5,5), np.float32)/25;
    conv_img = cv2.filter2D(inp_img, -1, kernel);
    
    mask = cv2.addWeighted(inp_img, 1, conv_img, -1, 0);
    
    out_img = cv2.addWeighted(inp_img, 1, mask, 1, 0);
    cv2.imshow(out_img);
    #return out_img;
    
def saltPepNoise(inp_img): #to denoise salt and pepper
    out_img = cv2.medianBlur(inp_img, 3);
    cv2.imshow(out_img);
    #return out_img;
    
