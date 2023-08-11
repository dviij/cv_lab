import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_specification(input_image, reference_image):
    input_hist, _ = np.histogram(input_image.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference_image.flatten(), 256, [0, 256])

    input_cdf = input_hist.cumsum()
    reference_cdf = reference_hist.cumsum()

    input_cdf_normalized = input_cdf * float(input_cdf.max()) / input_cdf[-1]
    reference_cdf_normalized = reference_cdf * float(input_cdf.max()) / reference_cdf[-1]

    matched_image = np.interp(input_cdf_normalized, reference_cdf_normalized, np.arange(256))

    matched_image = matched_image[input_image]

    return matched_image


if __name__ == "__main__":
    # Load the input and reference images
    input_image_path = "C:\\Users\\ugcse.PG-CP.000\\Desktop\\highres.jpg"
    reference_image_path = "C:\\Users\\ugcse.PG-CP.000\\Desktop\\reff.jpg"

    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    if input_image is None or reference_image is None:
        print("Error: Unable to load images.")
    else:
        # Perform histogram specification
        matched_image = histogram_specification(input_image, reference_image)

        # Calculate histograms for the images
        input_hist, _ = np.histogram(input_image.flatten(), 256, [0, 256])
        reference_hist, _ = np.histogram(reference_image.flatten(), 256, [0, 256])
        matched_hist, _ = np.histogram(matched_image.flatten(), 256, [0, 256])

        # Display the input, reference, matched images, and histograms
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 3, 1)
        plt.imshow(input_image, cmap='gray')
        plt.title("Input Image")

        plt.subplot(2, 3, 2)
        plt.imshow(reference_image, cmap='gray')
        plt.title("Reference Image")

        plt.subplot(2, 3, 3)
        plt.imshow(matched_image, cmap='gray')
        plt.title("Matched Image")

        plt.subplot(2, 3, 4)
        plt.plot(input_hist, color='r')
        plt.title("Input Histogram")

        plt.subplot(2, 3, 5)
        plt.plot(reference_hist, color='g')
        plt.title("Reference Histogram")

        plt.subplot(2, 3, 6)
        plt.plot(matched_hist, color='b')
        plt.title("Matched Histogram")

        plt.tight_layout()
        plt.show()
