import sys
import cv2
import numpy as np


start_point = (0, 0)
end_point = (0, 0)

def draw_rectangle(image, start_point, end_point, color=(255, 0, 0), thickness=2):
    return cv2.rectangle(image, start_point, end_point, color, thickness)

def mouse_drawing(event, x, y, flags, params):
    global start_point, end_point
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        end_point = (x+416, y+416)
        print(start_point, end_point)

image_org = cv2.imread(sys.argv[1]);

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)

image = image_org
while True:
    if start_point != end_point:
        pass
        #draw_image = draw_rectangle(image, start_point, end_point)
        #image = draw_image

    cv2.imshow( "Frame", image);
    key = cv2.waitKey(0)

    if key == 27:
        break
    elif key == ord("d"):
        image = image_org
        start_point = (0, 0)
        end_point = (0, 0)

cv2.destroyAllWindows()
