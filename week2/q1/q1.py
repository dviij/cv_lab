import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('C:\\Users\\ugcse.PG-CP.000\\Desktop\\highres.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

# Perform histogram equalization
equalized_img = cv.equalizeHist(img)

# Display the original image, histogram, and equalized image side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 3)
plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.title("Histogram")

plt.subplot(1, 3, 2)
plt.imshow(equalized_img, cmap='gray')
plt.title("Equalized Image")

plt.tight_layout()
plt.show()

