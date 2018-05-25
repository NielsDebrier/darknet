#!/usr/bin/env python
from ctypes import *
import math
import random
import cv2
import rospy
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ros_yolov3.msg import Detection, Object


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/tim/stage/catkin_ws/src/ros_yolov3/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, np_img, thresh=.5, hier_thresh=.5, nms=.45):
    # im = load_image(image, 0, 0)
    im = nparray_to_image(np_img)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def nparray_to_image(img):
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)

    return image


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


class ObjectDetection:
    def __init__(self):
        # initiaze yolo
        dirname = os.path.dirname(__file__)
        if rospy.has_param('~config') and rospy.has_param('~weights') and rospy.has_param('~meta'):
            config = os.path.join(dirname, rospy.get_param('~config'))
            weights = os.path.join(dirname, rospy.get_param('~weights'))
            meta = os.path.join(dirname, rospy.get_param('~meta'))
        else:
            config = os.path.join(dirname, 'cfg/yolov3.cfg')
            weights = os.path.join(dirname, 'bin/yolov3.weights')
            meta = os.path.join(dirname, 'cfg/coco.data')
        self.net = load_net(config, weights, 0)
        self.meta = load_meta(meta)

        #opencv bridge
        self.bridge = CvBridge()

        #fps counter
        
        
        # publisher and subscribers
        self.annotation_publisher = rospy.Publisher('yolo_annotation', Detection, queue_size=1)
        self.image_publisher = rospy.Publisher('yolo_image', Image, queue_size=1)
        if rospy.has_param('~source'):
            topic_name = rospy.get_param('~source')
        else:
            topic_name = '/usb_cam/image_raw'
        self.subscriber = rospy.Subscriber(topic_name, Image, self.callback, queue_size=1, buff_size=2 ** 24)
        
    def callback(self, image_message):
        try:
            image = self.bridge.imgmsg_to_cv2(image_message, 'bgr8')
        except CvBridgeError as e:
            print(e)

        
        msg_detection = Detection()
        r = detect(self.net, self.meta, image)
        for i in r:
            msg_object = Object()
            x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
            name = i[0].decode()
            confidence = str(round(i[1] * 100, 2))
            cv2.putText(image, name + " [" + confidence + "]", (pt1[0], pt1[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 4)

            msg_object.object_name = str(name)
            msg_object.confidence = float(confidence)
            msg_object.x_min = xmin
            msg_object.y_min = ymin
            msg_object.x_max = xmax
            msg_object.y_max = ymax
            msg_detection.object_list.append(msg_object)

        #cv2.imshow("img", image)
        self.annotation_publisher.publish(msg_detection)
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))

        
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            exit()


def main():
    rospy.init_node('yolo_node', anonymous=False)
    ObjectDetection()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS object detection with YOLOv3"
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()