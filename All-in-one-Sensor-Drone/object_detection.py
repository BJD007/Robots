import cv2
import numpy as np

def detect_falling_objects(camera):
    ret, frame = camera.read()
    if not ret:
        return False

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # Store the first frame for comparison
    if not hasattr(detect_falling_objects, "first_frame"):
        detect_falling_objects.first_frame = blurred
        return False

    # Compute absolute difference between current frame and first frame
    frame_delta = cv2.absdiff(detect_falling_objects.first_frame, blurred)

    # Apply threshold to highlight differences
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours on thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small contours
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Check if the contour is moving downwards
        if not hasattr(detect_falling_objects, "prev_y"):
            detect_falling_objects.prev_y = y
        elif y > detect_falling_objects.prev_y:
            detect_falling_objects.prev_y = y
            return True  # Falling object detected

    detect_falling_objects.prev_y = None
    return False
