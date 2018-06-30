# Darknet combined with ROS

This repository is a result of the internship from the student Niels Debrier with the company Craftworkz in cooperation with PXL University College. It is a fork of the repository darknet wich is used for the object detection algorithm You Only Look Once (YOLO).

This repository expands the usability of the yolo algorithm. We added ROS integration so that we can detect objects in a camera stream from robots or other devices which uses ROS.

## Installation
There are two ways to install and use this repository. The first one is by downloading it yourself. The second one is by using a docker container.

### Install
1. CD into your catkin workspace (`$ cd ~/catkin_ws/src`)
  * if you don't have a catkin workspace, follow tutorial: [here](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)
Clone the repository using the following command: \
`$ git clone https://github.com/PXLRoboticsLab/darknet.git` 

2. When you have a GPU:
- Go to the folder with the commandline
- Go into the folder `darknet`
- Use the following command: `$ make`

When you don't have a GPU:
- Go to the folder with the commandline
- Go into the folder `darknet`
- Open the Makefile and change GPU=1 and cuDNN=1 to GPU=0 and cuDNN=0.
- Use the following command: `$ make`

3. Download the weights file for YOLOv3 into cfg. You can download the weights file [here](https://pjreddie.com/media/files/yolov3.weights). Move this into darknet > cfg folder.

4. Download a ROS node that has a camera feed like the [pepper_robot](https://github.com/PXLRoboticsLab/ROS_Pepper/blob/master/ROS-Pepper.md) or [usb_cam](https://github.com/ros-drivers/usb_cam). 

5. You can now start the program with following command:
`$ roslaunch ros_yolo yolo.launch`

This will start the usb_cam with the YOLOv3 algorithm.

### Dockerfile
In the folder docker you will find a file which you can build with docker.
If you have a GPU then eveything is set and can be used immediately because all the necessary dependencies are inside the docker container.
