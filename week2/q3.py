import cv2
img=cv2.imread("C:\\Users\\ugcse.PG-CP.000\\Desktop\\highres.jpg")
new= cv2.resize(img,(500,200))
crop=img[0:500,0:500]
cv2.imshow("Original",img)
cv2.imshow("Cropped",crop)
cv2.imshow("Resized",new)
cv2.waitKey(0)
cv2.destroyAllWindows()
