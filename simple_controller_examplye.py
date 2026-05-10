#simple controller with onboard camera

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

#configuration constants
DEBOUNCE_TIME = 0.1 #100 milliseconds
MAX_ANGLE = 0.5
MAX_SPEED = 250
SPEED_INCR = 5
ANGLE_INCR = 0.05

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing example
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img


#Display image on onboard display
def display_image(display, image):
    display.setColor(0x000000)
    display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)

## Function to get the region of interest (ROI)s
def region_of_interest(edges):
    height, width = edges.shape

    ##Definition of the vertices to filter region of interest. 
    vertices = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.35), int(height * 0.6)),
        (int(width * 0.65), int(height * 0.6)),
        (int(width * 0.90), height)
    ]], dtype=np.int32)

    ##Appply mask to the edges image.
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)

    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def hough_lines(roi_edges): 
    lines = cv2.HoughLinesP(
        roi_edges,
        1,
        np.pi / 180,
        20,
        minLineLength = 20, 
        maxLineGap=15
    )
    return lines

def draw_lines(image, lines): 
    line_image = np.zeros_like(image)
    if lines is not None: 
        for line in lines: 
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return line_image

## Function to compute the center of the lanes with the 
## lanes as a refence
def compute_lane_center(lines): 

    ## Check to confirm there are lines to 
    ## calculate from
    if lines is None: 
        return None
    
    ## Variables to save left and right "points"
    ## to steer. 
    left_points = []
    right_points = []
    all_points = []

    ## Loop for every line detected
    for line in lines: 
        ## Retrieve the line element values correclty. 
        x1, y1, x2, y2 = line[0]

        ## Skips vertical lines avoiding division by 0 in the slope
        ## calculation later. 
        if x2 == x1: 
            continue

        ## Calculation of slope
        slope = (y2-y1) / (x2 - x1)

        ## Insignificant change in slope. 
        if abs(slope) < .3: 
            continue
        
        
        all_points.extend([x1, x2])

        if slope < 0: 
            left_points.extend([x1,x2])
        else: 
            right_points.extend([x1, x2])

    if left_points and right_points:
        left_x = np.mean(left_points)
        right_x = np.mean(right_points)
        return (left_x + right_x) / 2.0
    
    if all_points:
        return np.mean(all_points)

    return None


# main
def main():
    speed = 50
    last_press = {}
    kp = 0.35
    ki = 0.0
    kd = 0.05
    integral = 0.0
    previous_error = 0.0
    previous_time = time.time()
    steering = 0.0

    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    # display_img = Display("display_image")
    display_img = robot.getDevice("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)
        
        display_w = display_img.getWidth()
        display_h = display_img.getHeight()

        # Webots camera returns BGRA; convert before grayscale/Canny.
        bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        resized_bgr = cv2.resize(bgr_image, (display_w, display_h))
        grey_image = greyscale_cv2(resized_bgr)
        grey_resized = grey_image

        ## Edge detection
        canny = cv2.Canny(grey_resized, 50, 150)

        ## Roi edges
        roi_edges = region_of_interest(canny)

        ##Hough lines
        lines = hough_lines(roi_edges)
        lane_center_x = compute_lane_center(lines, display_w)

        current_time = time.time()
        dt = current_time - previous_time
        if dt <= 0:
            dt = 1e-3

        if lane_center_x is not None:
            image_center_x = display_w / 2.0
            error = (lane_center_x - image_center_x) / image_center_x
            integral += error * dt
            integral = max(-1.0, min(1.0, integral))
            derivative = (error - previous_error) / dt

            steering = kp * error + ki * integral + kd * derivative
            steering = max(-MAX_ANGLE, min(MAX_ANGLE, steering))

            previous_error = error
        else:
            integral = 0.0
            steering = 0.0

        previous_time = current_time

        line_image = draw_lines(np.zeros((display_h, display_w, 3), dtype=np.uint8), lines)

        ## Display final image
        line_image_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        debug_view = cv2.addWeighted(roi_edges, 0.7, line_image_gray, 1.0, 0)
        display_image(display_img, debug_view)

        #to reduce rebounds
        # Read keyboard
        key=keyboard.getKey()

        if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
            continue # Ignore rebound

        #pressed key accepted, update
        last_press[key] = current_time

        if key == ord('A'):
            #filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
            
        #update angle and speed
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
