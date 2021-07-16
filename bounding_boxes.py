import cv2
from facenet_models import FacenetModel
import matplotlib.pyplot as plt
from camera import take_picture
from matplotlib.patches import Rectangle

fig,ax = plt.subplots()
pic = take_picture()
# ax.imshow(pic)

model = FacenetModel()

boxes, probabilities, landmarks = model.detect(pic)


for box, prob, landmark in zip(boxes, probabilities, landmarks):
    # draw the box on the screen
    plt.text(box[0], box[1], 'box_label', bbox=dict(facecolor='blue', alpha=0.5))
    ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))