import cv2
img=cv2.imread('C:\\Users\\ugcse.PG-CP.000\\PycharmProjects\\210962046\\files\\test_car.jpg')
box_fill=cv2.boxFilter(img,-1,(3,3))
gauss_fill=cv2.GaussianBlur(img,(3,3),0)
cv2.imshow("original image",img)
cv2.imshow("Gaussian Filter",gauss_fill)
cv2.imshow("Box Filter",box_fill)
cv2.waitKey(0)
cv2.destroyAllWindows()