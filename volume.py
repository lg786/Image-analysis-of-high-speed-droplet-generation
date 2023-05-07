import cv2
import numpy as np
import csv

def detect_circles(image_path, circle_radius):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                               param1=50, param2=30, minRadius=int(circle_radius*0.8),
                               maxRadius=int(circle_radius*1.2))

    # If circles were detected, draw them on the image and store their diameters in a CSV file
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        with open('circle_diameters.csv', mode='w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Index', 'Diameter'])
            for i, (x, y, r) in enumerate(circles):
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                diameter = r * 2
                writer.writerow([i, diameter])

    return image, circles

image_path = "/home/lahari/Microfluidics Project/frames/1246.jpg"
circle_radius = 20

result_image, circles = detect_circles(image_path, circle_radius)
cv2.imwrite("result.jpg", result_image)
