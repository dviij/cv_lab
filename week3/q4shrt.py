import cv2
import numpy as np

img = cv2.imread("C:\\Users\\ugcse.PG-CP.000\\PycharmProjects\\210962046\\files\\lenna.jpg", cv2.IMREAD_GRAYSCALE)

# Create the Laplacian edge-detected image
lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

# Stack the images horizontally
stacked_images = np.hstack((img, lap))

# Display the stacked images
cv2.imshow("Original Image vs. Laplacian", stacked_images)

cv2.waitKey(0)
cv2.destroyAllWindows()
