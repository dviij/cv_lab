import cv2
import numpy as np
def compute_gradient(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients in x and y directions using Sobel operators
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_x, gradient_y, gradient_magnitude, gradient_direction


if __name__ == "__main__":
    # Replace with the actual path to your image
    image_path = "C:\\Users\\ugcse.PG-CP.000\\PycharmProjects\\210962046\\files\\lenna_1.jpg"

    # Read the original image
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error: Unable to read image.")
    else:
        # Compute gradient information using the defined function
        gradient_x, gradient_y, gradient_magnitude, gradient_direction = compute_gradient(original_image)

        # Display the original image and different gradient components
        cv2.imshow("Original Image", original_image)
        cv2.imshow("Gradient X", cv2.convertScaleAbs(gradient_x))
        cv2.imshow("Gradient Y", cv2.convertScaleAbs(gradient_y))
        cv2.imshow("Gradient Magnitude", cv2.convertScaleAbs(gradient_magnitude))
        cv2.imshow("Gradient Direction", cv2.convertScaleAbs((gradient_direction + np.pi) * 255 / (2 * np.pi)))

        cv2.waitKey(0)
        cv2.destroyAllWindows()
