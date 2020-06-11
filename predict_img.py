import sys
sys.path.insert(1, '/Users/zhenningyang/Documents/opencv_moving_box/src')
from darknet_func import *

if __name__ == "__main__":

    # load model
    net = load_net(b"/Users/zhenningyang/Documents/darknet/cfg/yolov3_ssig.cfg",
                   b"/Users/zhenningyang/Documents/yolov3_weights/yolov3_ssig_final.weights", 0)
    meta = load_meta(b"/Users/zhenningyang/Documents/opencv_moving_box/ssig.data")

    r = detect(net, meta, bytes(sys.argv[1], 'utf-8'))
    count = len(r)
    img = draw("{}".format(sys.argv[1]), r)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display Prediction
    cv2.imshow('prediction',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('prediction.png',img)
        cv2.destroyAllWindows()
