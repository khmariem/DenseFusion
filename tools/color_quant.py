import numpy as np
import cv2 as cv

def quant(file):
    img = cv.imread(file)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2
    # #print(label)

    # cv.imshow('res2',res2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()