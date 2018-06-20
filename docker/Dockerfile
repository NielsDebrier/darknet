FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# install ros
RUN apt-get update \
        && apt-get install -y --no-install-recommends ros-kinetic-desktop-full \
		&& rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# environment setup
RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# opencv & git & wget
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        libopencv-dev \
        git \
        wget

# download usb_cam and yolo
RUN mkdir -p /root/catkin_ws/src \
    && git clone https://github.com/bosch-ros-pkg/usb_cam.git /root/catkin_ws/src/usb_cam \
    && git clone -b docker_stable https://github.com/NielsDebrier/darknet.git /root/catkin_ws/src/ros_yolov3 \
    && make -C /root/catkin_ws/src/ros_yolov3/darknet/ \
    && wget -P /root/catkin_ws/src/ros_yolov3/darknet/bin/ https://pjreddie.com/media/files/yolov3.weights \
    && /bin/bash -c ". /opt/ros/kinetic/setup.bash; cd /root/catkin_ws; catkin_make" \
    && /bin/bash -c "source /root/catkin_ws/devel/setup.bash"

RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc