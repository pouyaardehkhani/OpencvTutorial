from matplotlib import pyplot as plt
from easyocr import Reader
import pandas as pd
import cv2
import time

# Doing some Face Recognition with the webcam:
# We turn the webcam on.
video_capture = cv2.VideoCapture(0)
# 0 if it's the webcam of the computer
# 1 if it's an external device

# We repeat infinitely (until break):
while True: 
    # We get the last frame.
    _, frame = video_capture.read() 
    # OCR the input image using EasyOCR
    reader = Reader(['en'], gpu = True)
    results = reader.readtext(frame)
    
    all_text = []

    # iterate over our extracted text 
    for (bbox, text, prob) in results:
        # display the OCR'd text and the associated probability of it being text
        print(f" Probability of Text: {prob*100:.3f}% OCR'd Text: {text}")

        # get the bounding box coordinates
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        # Remove non-ASCII characters from the text so that
        # we can draw the box surrounding the text overlaid onto the original image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        all_text.append(text)
        cv2.rectangle(frame, tl, br, (255, 0, 0), 2)
        cv2.putText(frame, text, (tl[0], tl[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # We display the outputs.
    cv2.imshow('Video', frame) # show the result in animated window
    # If we type on the keyboard: (press q)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        # We stop the loop.
        break 

# We turn the webcam off.
video_capture.release() 
# We destroy all the windows inside which the images were displayed.
cv2.destroyAllWindows()  