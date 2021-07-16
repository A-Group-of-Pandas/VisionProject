import cv2
from detector import Detector
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk


# def face_detection():
# video capture
capture = cv2.VideoCapture(0)

# video loop
while True:
    frame_status, img = capture.read()
    cv2.imshow('cam_feed', img)

    if cv2.waitKey(20) & 0xFF == ord(' '):
        break

cv2.imwrite('image.jpg', img)