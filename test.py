import cv2

capture = cv2.VideoCapture(0)
ret, img = capture.read()
print(type(img))
box = [200.0,300.0,100.0,400.0]

img = cv2.rectangle(img, (200,100), (400, 500), (36,255,12), 1)
cv2.putText(img, 'hello', (200, 100-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # plt.text(box[0], box[1], name, bbox=dict(facecolor='blue', alpha=0.5))
        # ax.add_patch(Rectangle(box[:2], *(box[2:] - box[:2]), fill=None, lw=2, color="red"))
    # cv2.rectangle()
cv2.imshow("name",img)
#plt.show()
cv2.waitKey()