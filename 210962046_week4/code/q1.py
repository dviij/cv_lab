import cv2
originalImage = cv2.imread("c://users//ugcse.PG-CP.000//PycharmProjects//210962046//files//lenna_1.jpg", cv2.IMREAD_GRAYSCALE)

_, binImage1 = cv2.threshold(originalImage, 127, 255,  cv2.THRESH_OTSU)
_, binImage2 = cv2.threshold(originalImage, 127, 255,  cv2.THRESH_BINARY)
_, binImage3 = cv2.threshold(originalImage, 127, 255,  cv2.THRESH_TRUNC)
_, binImage4 = cv2.threshold(originalImage, 127, 255,  cv2.THRESH_BINARY_INV+cv2.THRESH_MASK)
_, binImage5 = cv2.threshold(originalImage, 127, 255,  cv2.THRESH_BINARY+cv2.THRESH_MASK+cv2.THRESH_OTSU+cv2.THRESH_TRUNC+cv2.THRESH_OTSU+cv2.THRESH_TRIANGLE)
cv2.imshow('original image', originalImage)
cv2.imshow('threshold image1', binImage1)
cv2.imshow('threshold image2', binImage2)
cv2.imshow('threshold image3', binImage3)
cv2.imshow('threshold image4', binImage4)
cv2.imshow('threshold image5', binImage5)
cv2.waitKey(0)
cv2.destroyAllWindows()
