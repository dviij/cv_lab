import cv2

# Replace with the path to your image
image_path = "path_to_your_image.jpg"

# Read the image in grayscale
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    print("Error: Unable to read image.")
else:
    # Define low and high thresholds for Canny
    low_threshold = 50
    high_threshold = 150

    # Apply Gaussian blur to reduce noise and then Canny edge detection
    blurred = cv2.GaussianBlur(original_image, (5, 5), 0)
    edge_image = cv2.Canny(blurred, low_threshold, high_threshold)

    # Display the images
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Canny Edge Detection", edge_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
