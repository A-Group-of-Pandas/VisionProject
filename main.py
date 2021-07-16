import cv2
from detector import Detector
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from detector import Detector
from database import Database
import camera
from matplotlib.patches import Rectangle
import time
import numpy as np
from profile import Profile

model = Detector()
database = Database()
fig, ax = plt.subplots()
database.load('database.pkl')

def detect_live():
    #plt.ion()
    fig, ax = plt.subplots()
    capture = cv2.VideoCapture(0)
    #show_image = ax.imshow(capture.read()[1])
    while True:
        ret, img = capture.read()
        if not ret:
            continue
        img = img[:,:,::-1]
        detect_and_draw(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def detect_once():
    capture = camera.take_picture()
    unknown = detect_and_draw(capture)
    unknownCase(capture,unknown)

def draw_bounding_box(img,name,box):
    img = cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), (36,255,12), 1)
    cv2.putText(img, name, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.002*img.shape[0], color=(36,255,12), thickness=1)
    return img


def detect_and_draw(img):
    boxes = model.detect(img)
    vectors = model.get_vectors(img,boxes)
    img = img[:,:,::-1]/255.0
    unknown = []
    for i,(box,vector) in enumerate(zip(boxes,vectors)):
        name = database.search(vector)
        print(name)
        # draw the box on the screen
        if name is None:
            unknown.append((box,vector))
            name = "Unknown"
        
        img = draw_bounding_box(img,name,box)
        #print("Bounding box size: ", box[2]-box[0], box[3]-box[1])
        # height = box[2]-box[0]
        # width = box[3]-box[1]

        # if height*width >= 0.15*img.size:
        #cv2.imshow('output',img)
        #print(img.shape)
        cv2.imshow('output',img)
    return unknown

def unknownCase(img,unknown):
    for box,vector in unknown:
        img2 = draw_bounding_box(img.copy(),"who is this?",box)
        cv2.imshow('query',img2)
        newname = input("Please insert new profile name (or none if not a face): ")
        if newname == "none":
            continue
        database.add(Profile(newname, vector))
        print("Success: new profile [ " + newname + " ] added")

def detect_from_file(img_path):
    img = cv2.imread(img_path)
    print(img)
    img2 = img[:,:,::-1].copy()
    unknown = detect_and_draw(img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    unknownCase(img,unknown)

print("Choose A Detection Method:")
print("(1) Detect Live")
print("(2) Detect From File")

detect_method = int(input())
if detect_method == 1:
    detect_live()
elif detect_method == 2:
    file_name = input('Enter the file name:')
    detect_from_file(file_name)





