import cv2
img=cv2.imread("c://users//ugcse.PG-CP.000//PycharmProjects//210962046//files//lenna_1.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lb = (0, 40, 50)
ud = (200, 255, 255)

mask = cv2.inRange(hsv, lb, ud)
segmented_image = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('segmented_image.jpg', segmented_image)