import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load the RGB image
img = cv.imread('C:\\Users\\ugcse.PG-CP.000\\Desktop\\highres.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"

# Split the image into color channels
b, g, r = cv.split(img)

# Perform histogram equalization on each channel
b_eq = cv.equalizeHist(b)
g_eq = cv.equalizeHist(g)
r_eq = cv.equalizeHist(r)

# Merge the equalized channels back into an RGB image
equalized_img = cv.merge((b_eq, g_eq, r_eq))

# Display the original image, histogram, and equalized image side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original Image")

# Display histograms for each color channel
for i, color in enumerate(['r', 'g', 'b']):
    hist, bins = np.histogram(img[:,:,i].flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.subplot(1, 3, 2)
    plt.plot(cdf_normalized, color=color)
    plt.hist(img[:,:,i].flatten(), 256, [0, 256], color=color, alpha=0.5)
    plt.xlim([0, 256])

plt.legend(['r', 'g', 'b'], loc='upper left')
plt.title("Histograms")

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(equalized_img, cv.COLOR_BGR2RGB))
plt.title("Equalized Image")

plt.tight_layout()
plt.show()
