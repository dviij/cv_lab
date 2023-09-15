import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('C://Users//ugcse.PG-CP.000//PycharmProjects//210962046//files//lenna.jpg', cv.IMREAD_GRAYSCALE) # `<opencv_root>/samples/data/blox.jpg`
# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(220,0,200))
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

img3 = cv.drawKeypoints(img, kp, None, color=(10,255,50))

# plt.figure(figsize=(10,6))

plt.subplot(221)
plt.imshow(img,cmap='gray')
plt.title("original image")

plt.subplot(222)
plt.imshow(img2,cmap='gray')
plt.title("fast true")

plt.subplot(223)
plt.imshow(img3,cmap='gray')
plt.title("fast false")
plt.tight_layout()
plt.show()