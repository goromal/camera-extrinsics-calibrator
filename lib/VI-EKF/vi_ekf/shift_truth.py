#!/usr/bin/env python

import rosbag
import rospy
from tqdm import tqdm
import sys

outbag = rosbag.Bag('/home/superjax/rosbag/with_yaw_and_truth_shifted.bag', 'w')
inbag = rosbag.Bag('/home/superjax/rosbag/with_yaw_and_truth.bag')

truth_offset = 1523486528.443879843 - 1220.81666
camera_offset = 0.03

for topic, msg, t in tqdm(inbag.read_messages(), total=inbag.get_message_count()):
    if topic == '/vrpn/Leo/pose':
        outbag.write('/truth/pose', msg, msg.header.stamp + rospy.Duration.from_sec(truth_offset))
    elif topic == '/imu':
        outbag.write('/imu', msg, msg.header.stamp)
    elif topic == '/camera/color/image_raw':
    	outbag.write('/color', msg, msg.header.stamp + rospy.Duration.from_sec(camera_offset))
    elif topic == '/camera/depth/image_rect_raw':
    	outbag.write('/depth', msg, msg.header.stamp + rospy.Duration.from_sec(camera_offset))
    else:
    	outbag.write(topic, msg, t)

print "bag parsing done! - reindexing"

outbag.reindex()

print "complete"