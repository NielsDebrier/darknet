#!/usr/bin/env python
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class Subscriber_node:
    def __init__(self):
        self.counter = 0
        self.x = 1
        self.start_time = time.time()
        
        rospy.Subscriber("yolo_image", Image, self.callback, queue_size=1, buff_size=2 ** 24)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def callback(self, image_message):
        bridge = CvBridge()
        try:
            image = bridge.imgmsg_to_cv2(image_message, 'bgr8')
        except CvBridgeError as e:
            print(e)

        cv2.imshow("img", image)
        self.counter += 1
        if (time.time() - self.start_time) > self.x:
            print("FPS: ", self.counter / (time.time() - self.start_time))
            self.counter = 0
            self.start_time = time.time()
        cv2.waitKey(1)
        

def main():
    rospy.init_node('yolo_listener', anonymous=True)
    Subscriber_node()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS object detection with YOLOv3"
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()