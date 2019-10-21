import cv2 as cv

def selectme(path):

    im = cv.imread(path)
    r = cv.selectROI(im)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return r