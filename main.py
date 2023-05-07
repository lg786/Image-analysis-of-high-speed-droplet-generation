import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt

fps = int(input("Enter the frames per second value for the slowed down input video: " ))
ofps = int(input("Enter the frames per second value for the original input video: " ))

def extract_frames(video_path, output_dir, fps):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize a frame counter
    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Construct the filename for this frame
        filename = f"{frame_count:04d}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Save the frame to the output directory
        cv2.imwrite(filepath, frame)

        # Increment the frame counter
        frame_count += 1

        # Set the frame rate of the video capture object
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * fps)

    # Release the video capture object
    cap.release()

extract_frames("/home/lahari/Microfluidics Project/beadcells.mp4", "/home/lahari/Microfluidics Project/frames", fps)

pipe_radius = int(input("Enter the radius of the pipe in micrometers"))

def detect_circles(frames_dir, circle_radius_micrometers, output_filename, pixels_per_micrometer, fps):
    # Get the list of frame filenames from the output directory of extract_frames
    frame_filenames = sorted(os.listdir(frames_dir))

    # Initialize the list of detected circles
    circles_list = []

    # Load the first frame to get the frame size
    first_frame_path = os.path.join(frames_dir, frame_filenames[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Loop over the frames
    for filename in frame_filenames:
        # Load the frame
        frame_path = os.path.join(frames_dir, filename)
        frame = cv2.imread(frame_path)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles in the image
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                   param1=50, param2=30, minRadius=int(circle_radius_micrometers*pixels_per_micrometer*0.8),
                                   maxRadius=int(circle_radius_micrometers*pixels_per_micrometer*1.2))

        # If circles were detected, draw them on the frame and add them to the list
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                r_in_pixels = int(r/pixels_per_micrometer)
                cv2.circle(frame, (x, y), r_in_pixels, (0, 255, 0), 2)
                circles_list.append((filename, x, y, r_in_pixels*pixels_per_micrometer))

        # Write the frame to the output video
        output_video.write(frame)

    # Release the output video and return the list of detected circles
    output_video.release()
    return circles_list

circles = detect_circles("/home/lahari/Microfluidics Project/frames", pipe_radius, "output_video.mp4", 0.8, fps)


def calculate_average_speed(circles_list, pixel_size):
    # Initialize the list of speeds
    speeds = []

    # Loop over the circles, skipping the first frame
    for i in range(1, len(circles_list)):
        # Get the circle and filename for this frame and the previous frame
        circle1 = circles_list[i]
        circle2 = circles_list[i-1]
        filename1, x1, y1, r1 = circle1
        filename2, x2, y2, r2 = circle2

        # Check if the current and previous filenames are the same
        if filename1 == filename2:
            continue

        # Calculate the distance traveled between the two frames in pixels
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Calculate the time elapsed between the two frames in seconds
        time_elapsed = (int(filename1[:-4]) - int(filename2[:-4])) / fps

        # Calculate the speed in pixels per second
        speed = distance / time_elapsed

        # Convert speed to pixels per micron
        speed = speed * pixel_size

        # Add the speed to the list
        speeds.append(speed)

    # Calculate the average speed
    avg_speed = np.mean(speeds)

    return avg_speed

avg_speed_of_droplets = (calculate_average_speed(circles, 0.8)*(ofps/fps))/1000000
print("The average speed of the droplets", avg_speed_of_droplets, "m/s")




'''def find_avg_droplet_radius(circle_list, pixel_size, circle_radius):
    # Initialize the list of detected droplet diameters
    diameters = []

    # Loop over the circles
    for circle in circle_list:
        # Get the filename and circle parameters
        filename, x, y, r = circle

        

        # Calculate the coordinates of the circle center in pixels
        x_pixels = int(round(x / pixel_size))
        y_pixels = int(round(y / pixel_size))

        # Calculate the radius of the circle in pixels
        r_pixels = int(round(r / pixel_size))

        # Load the frame
        frame = cv2.imread(filename)

        print("haw")

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles in the image
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                   param1=50, param2=30, minRadius=int(circle_radius*0.8/pixel_size),
                                   maxRadius=int(circle_radius*1.2/pixel_size), 
                                   center=(x_pixels, y_pixels))

        # If circles were detected, calculate the diameter of each droplet and add
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x_pix, y_pix, r_pix) in circles:
                diameter = r_pix * 2 * pixel_size
                diameters.append(diameter)

    # Calculate the average diameter
    avg_diameter = np.mean(diameters)

    # Calculate the average radius
    avg_radius = avg_diameter / 2

    # Calculate the standard deviation of the radius
    std_radius = np.std(diameters) / 2

    # Return the average radius and standard deviation of the detected droplets
    return avg_radius, std_radius


avg_radius, std_radius = find_avg_droplet_radius(circles, 0.8, pipe_radius)
print("The avg droplet radius is", avg_radius, "micrometers")
print("The standard deviation of droplet radius is", std_radius, "micrometers")'''



def plot_histogram(circles_list, pixel_size):
    volumes = []
    for circle in circles_list:
        r = circle[3]
        volume = (4/3) * np.pi * (r ** 3)
        volume = volume / (pixel_size ** 3)
        volumes.append(volume)

    plt.hist(volumes, bins=20)
    plt.xlabel('Volume of drop (in microns^3)')
    plt.ylabel('Number of drops')
    plt.title('Histogram of Volume vs Number of Drops')
    plt.show()

def plot_diameter_boxplot(circles_list, pixels_per_micrometer):
    # Initialize the list of droplet diameters
    diameters = []

    # Get the diameter of each droplet in micrometers
    for circle in circles_list:
        diameter = circle[3] * 2 / pixels_per_micrometer
        diameters.append(diameter)

    # Take a random sample of 20 droplets
    sample = np.random.choice(diameters, size=20, replace=False)

    # Create a box plot
    plt.boxplot(sample)
    plt.xlabel('Droplet Diameter (micrometers)')
    plt.title('Box Plot of Diameter for 20 Random Droplets')
    plt.show()


plot_histogram(circles, 0.8)
plot_diameter_boxplot(circles, 0.8)

def calculate_new_droplet_generation_speed(circles_list, pixel_size):
    # Initialize the list of new droplet generation speeds
    new_droplet_generation_speeds = []

    # Loop over the circles, skipping the first frame
    for i in range(1, len(circles_list)):
        # Get the circle and filename for this frame and the previous frame
        circle1 = circles_list[i]
        circle2 = circles_list[i-1]
        filename1, x1, y1, r1 = circle1
        filename2, x2, y2, r2 = circle2

        # Calculate the distance between the two circles in pixels
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        # If the distance between the two circles is greater than the sum of their radii,
        # it means that a new droplet has been generated
        if distance > (r1 + r2):
            # Calculate the time elapsed between the two frames in seconds
            time_elapsed = (int(filename1[:-4]) - int(filename2[:-4])) / 30.0

            # Calculate the new droplet generation speed in pixels per second
            new_droplet_generation_speed = distance / time_elapsed

            # Convert the new droplet generation speed to pixels per micron
            new_droplet_generation_speed = new_droplet_generation_speed / pixel_size

            # Add the new droplet generation speed to the list
            new_droplet_generation_speeds.append(new_droplet_generation_speed)

    # Calculate the average new droplet generation speed
    avg_new_droplet_generation_speed = np.mean(new_droplet_generation_speeds)

    return avg_new_droplet_generation_speed

def plot_droplet_distances(circles_list, pixel_size):
    # Initialize the list of detected droplet positions
    positions = []

    # Loop over the circles
    for circle in circles_list:
        # Get the filename and circle parameters
        filename, x, y, r = circle

        # Calculate the coordinates of the circle center in pixels
        x_pixels = int(round(x / pixel_size))
        y_pixels = int(round(y / pixel_size))

        # Add the position to the list
        positions.append((x_pixels, y_pixels))

    # Calculate the distances between all pairs of droplets
    distances = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            distance = np.sqrt(dx**2 + dy**2) * pixel_size
            distances.append(distance)

    # Calculate the average distance between droplets
    avg_distance = sum(distances) / len(distances)

    # Plot a histogram of the distances
    plt.hist(distances, bins='auto')
    plt.xlabel('Distance between droplets (µm)')
    plt.ylabel('Number of droplet pairs')
    plt.title('Distribution of droplet distances')
    plt.show()

    # Return the average distance between droplets
    return avg_distance

avg_distance = plot_droplet_distances(circles, 0.8)
print('Average distance between droplets:', avg_distance, 'µm')


