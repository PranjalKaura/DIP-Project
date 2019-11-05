import cv2
import numpy as np

in_img = cv2.imread('b.png')


def brighten(image, val):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    k = np.shape(v)
    print(k)
    for i in range(0, k[0]):
        for j in range(0, k[1]):
            if v[i, j] <= (255 - val):
                v[i, j] = v[i, j] + val

    fin = cv2.merge((h, s, v))
    out = cv2.cvtColor(fin, cv2.COLOR_HSV2BGR)
    return out


out_fin = brighten(in_img, 50)
cv2.imshow("Output", out_fin)
cv2.waitKey()