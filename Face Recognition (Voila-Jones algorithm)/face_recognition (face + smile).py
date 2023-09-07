# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades:
# We load the cascade for the face.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# We load the cascade for the eyes.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
# We load the cascade for the smile.
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections:
# We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
def detect(gray, frame): 
    # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    # second arg means the scale factor(how much the size of image is going to be reduce or increase) 
    # last arg is min number of neighbors
    
    # faces = tuple(x,y,width,height) 
    # x, y coordinates of the upper-left corner of the rectangle
    # width and height of the rectangle
    
    # For each detected face:
    for (x, y, w, h) in faces:
        # We paint a rectangle around the face.
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        # first arg is the image
        # second arg coordinate of the upper-left corner
        # third arg coordinate of the lower-right corner
        # forth arg color
        # fifth arg thickness of the edge of rectangles
        
        # we get two region of interest for detecting the eyes:
        # We get the region of interest in the black and white image.
        roi_gray = gray[y:y+h, x:x+w] 
        # We get the region of interest in the colored image.
        roi_color = frame[y:y+h, x:x+w] 
        
        # We apply the detectMultiScale method to locate smiles in the image.
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        # second arg means the scale factor(how much the size of image is going to be reduce or increase) 
        # last arg is min number of neighbors
        
        # smiles = tuple(x,y,width,height) 
        # x, y coordinates of the upper-left corner of the rectangle
        # width and height of the rectangle
        
        # For each detected smile:
        for (sx, sy, sw, sh) in smiles:
            # We paint a rectangle around the smile, but inside the referential of the face.
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            # first arg is the image
            # second arg coordinate of the upper-left corner
            # third arg coordinate of the lower-right corner
            # forth arg color
            # fifth arg thickness of the edge of rectangles
            
            
    # We return the image with the detector rectangles.
    return frame 

# Doing some Face Recognition with the webcam:
# We turn the webcam on.
video_capture = cv2.VideoCapture(0)
# 0 if it's the webcam of the computer
# 1 if it's an external device

# We repeat infinitely (until break):
while True: 
    # We get the last frame.
    _, frame = video_capture.read() 
    # We do some colour transformations.(convert to grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # first arg is the frame
    # second arg tells to do an average on blue, green and red and that's to take a scale of gray.
    # why average? to get the right contrast of black and white
    
    # We get the output of our detect function.
    canvas = detect(gray, frame) 
    # We display the outputs.
    cv2.imshow('Video', canvas) # show the result in animated window
    # If we type on the keyboard: (press q)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        # We stop the loop.
        break 

# We turn the webcam off.
video_capture.release() 
# We destroy all the windows inside which the images were displayed.
cv2.destroyAllWindows() 