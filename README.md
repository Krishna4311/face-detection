# Face detection using python

Real-Time Face Detection Using OpenCV. This project demonstrates real-time face detection using OpenCV in Python. The program uses a pre-trained Haar Cascade classifier to detect faces in a video stream captured from the default camera.

## Requirements

- Python 3.x
- OpenCV library (`opencv-python`)
- Haar Cascade XML file for frontal face detection

## Setup
1. **Install OpenCV**: If you haven't already, install the OpenCV library using pip:

    ```sh
    pip install opencv-python
    ```

2. **Download Haar Cascade XML File**: Download the Haar Cascade XML file for frontal face detection from [this link](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml) and save it in your project directory.


### Explanation

1. **Import Libraries**:
    ```python
    import cv2
    import numpy as np
    ```
    Import the necessary libraries: `cv2` for OpenCV functions and `numpy` for numerical operations.

2. **Load the Haar Cascade Classifier**:
    ```python
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    ```
    Load the Haar Cascade classifier for face detection using the path to the downloaded XML file.

3. **Define the Face Detection Function**:
    ```python
    def detect_faces(img):
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        # If no faces are detected, return the original image
        if faces is ():
            return img
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return img
    ```
    - Convert the input image to grayscale using `cv2.cvtColor`. Haar Cascade works better with grayscale images.
    - Detect faces in the grayscale image using `detectMultiScale`. This function returns a list of rectangles around detected faces.
    - If no faces are detected, return the original image.
    - Iterate through the list of detected faces and draw a blue rectangle around each face using `cv2.rectangle`.

4. **Initialize Video Capture**:
    ```python
    cap = cv2.VideoCapture(0)
    ```
    Initialize video capture from the default camera (index 0).

5. **Read Frames and Perform Face Detection**:
    ```python
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        # Detect faces in the frame
        frame = detect_faces(frame)
        
        # Display the frame with detected faces
        cv2.imshow('Video Face Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    ```
    - Continuously capture frames from the camera.
    - Call the `detect_faces` function to detect faces and draw rectangles around them.
    - Display the frame with detected faces in a window titled 'Video Face Detection'.
    - Break the loop and stop the video capture if the 'q' key is pressed.

6. **Release Resources**:
    ```python
    cap.release()
    cv2.destroyAllWindows()
    ```
    Release the video capture object and close all OpenCV windows.

This code will open a window displaying the video feed from your camera, with rectangles drawn around detected faces. Press 'q' to exit the program.

---
