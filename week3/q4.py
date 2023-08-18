import cv2
import numpy as np


def detect_edges_laplacian(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("grey image ",gray_image)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    return laplacian_abs


if __name__ == "__main__":
    image_path = "C:\\Users\\ugcse.PG-CP.000\\PycharmProjects\\210962046\\files\\highres.jpg"

    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error: Unable to read image.")
    else:
        edge_image = detect_edges_laplacian(original_image)

        # Stack the original and edge images horizontally for display
        stacked_image = np.hstack((original_image, cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)))

        cv2.imshow("Original Image vs. Laplacian Edge Detection", stacked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
