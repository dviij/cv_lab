import cv2
import numpy as np

def unsharp_masking(image, kernel_size=(5, 5), sigma=5.0, amount=4, threshold=10):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    cv2.imshow("burred_image",blurred)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    #sharpened = cv2.subtract(sharpened, cv2.multiply(sharpened, threshold / 255.0))
    return sharpened

if __name__ == "__main__":
    image_path = "C:\\Users\\ugcse.PG-CP.000\\PycharmProjects\\210962046\\files\\test_car.jpg"
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error: Unable to read image.")
    else:
        sharpened_image = unsharp_masking(original_image)

        cv2.imshow("Original Image", original_image)
        cv2.imshow("Sharpened Image", sharpened_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
