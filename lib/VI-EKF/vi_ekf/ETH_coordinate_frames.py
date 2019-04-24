import yaml
import numpy as np
import rospy
import rosbag
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from transform import *
import csv
import math
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
import scipy.misc
import scipy.interpolate
from os.path import isfile, join


bags = [
        'V1_01_easy',
        # 'V1_02_medium',
        # 'V1_03_difficult',
        # 'V2_01_easy',
        # 'V2_02_medium',
        # 'V2_03_difficult',
        # 'MH_01_easy',
        # 'MH_02_easy',
        # 'MH_03_medium',
        # 'MH_04_difficult',
        # 'MH_05_difficult'
]

duration = 30

for log in bags:
    yaml_file = open('/home/superjax/rosbag/EuRoC/' + log + '/mav0/cam0/sensor.yaml','r')
    T_IMU_C = Transform.from_T(np.reshape(np.array(yaml.load(yaml_file)['T_BS']['data']), (4,4)))
    yaml_file = open('/home/superjax/rosbag/EuRoC/' + log + '/mav0/vicon0/sensor.yaml','r')
    T_IMU_V = Transform.from_T(np.reshape(np.array(yaml.load(yaml_file)['T_BS']['data']), (4,4)))
    # Assume no translation between IMU and body
    T_IMU_V.translation = np.zeros((3,1))

    vicon_file = csv.reader(open('/home/superjax/rosbag/EuRoC/' + log + '/mav0/vicon0/data.csv','r'), delimiter=',')
    truth_file = csv.reader(open('/home/superjax/rosbag/EuRoC/' + log + '/mav0/state_groundtruth_estimate0/data.csv','r'), delimiter=',')
    imu_file = csv.reader(open('/home/superjax/rosbag/EuRoC/' + log + '/mav0/imu0/data.csv','r'), delimiter=',')
    images_dir = '/home/superjax/rosbag/EuRoC/'+ log + '/mav0/cam0/data'

    T_I_NWU = Transform.from_T(np.array([[1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, -1, 0,],
                                         [0, 0, 0, 1]]))

    outbag = rosbag.Bag('/home/superjax/rosbag/EuRoC/' + log + '_NED.bag', 'w')

    # load the groundtruth estimate
    truth = []
    for i, row in enumerate(truth_file):
        if i == 0: continue
        truth.append(row)
    truth = np.array(truth, dtype=np.float64)
    truth = scipy.interpolate.interp1d(truth[:,0], truth[:,1:].T, kind='cubic', bounds_error=False, fill_value="extrapolate")

    # Find the local-level frame by looking at the unbiased acceleration measurement
    count = 0
    acc = np.zeros((3,1))
    for i, row in enumerate(imu_file):
        if i == 0: continue
        if i > 800: break
        acc += np.array([row[4:]], dtype=np.float64).T + truth(float(row[0]))[-3:,None]
        count += 1
    acc /= count
    print norm(acc)
    acc /= norm(acc)
    # T_IMU_B = Transform(Quaternion.from_two_unit_vectors(np.array([[0, 0, -1]]).T, acc), np.array([[0, 0, 0]]).T)

    T_V_B = Transform.from_T(np.array([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0,],
                                       [0, 0, 0, 1]]))

    T_V_B.rotation += np.array([[-0.07, -0.09, 0.22]]).T

    T_IMU_B = T_IMU_V * T_V_B

    # Figure out the Vicon to Body Transform
    T_V_B = T_IMU_V.inverse * T_IMU_B
    print "T_B_V: ", T_V_B.inverse

    # Convert Camera Extrinsics to Proper Frame
    T_B_C = T_V_B.inverse* T_IMU_V.inverse * T_IMU_C
    print "T_B_C:", T_B_C

    # Convert IMU to proper frame
    # T_IMU_B = T_IMU_V * T_V_B
    imu = []
    t0 = 0
    T_I_B0 = Transform.Identity()
    tlist = []
    for i, row in enumerate(tqdm(imu_file)):
        if i == 0: continue
        if i == 1: t0 = int(row[0])
        time_ns = int(row[0])
        if (time_ns - t0) > duration * 1e9: break;

        groundtruth  = truth(float(row[0]))
        omega = np.array([row[1:4]], dtype=np.float64).T - groundtruth[-6:-3,None]
        acc = np.array([row[4:]], dtype=np.float64).T - groundtruth[-3:,None]

        acc = T_IMU_B.rotation.rotp(acc)
        omega = T_IMU_B.rotation.rotp(omega)

        # imu.append(np.vstack((omega, acc)).flatten().tolist())

        # Write the IMU message to the bag
        msg = Imu()
        msg.header.frame_id = "body"
        msg.header.stamp = rospy.Time(int(math.modf(time_ns/1e9)[1]), int(math.modf(time_ns/1e9)[0]*1e9))
        msg.angular_velocity.x = omega[0,0]
        msg.angular_velocity.y = omega[1,0]
        msg.angular_velocity.z = omega[2,0]
        msg.linear_acceleration.x = acc[0,0]
        msg.linear_acceleration.y = acc[1,0]
        msg.linear_acceleration.z = acc[2,0]
        outbag.write('imu', msg, msg.header.stamp)
        #
        # T_NWU_IMU = Transform(Quaternion(np.array([groundtruth[4:8]]).T), np.array([groundtruth[1:4]]).T)
        # T_I_B = T_I_NWU * T_NWU_IMU * T_IMU_B
        # if i == 1:
        #     T_I_B0 = Transform(Quaternion.from_euler(0, 0, T_I_B.rotation.euler[2,0]), T_I_B.translation)
        #
        # # Write the Associated Truth message to the bag
        # T_B0_B = T_I_B0.inverse * T_I_B
        # # tlist.append(np.vstack((T_B0_B.rotation.arr, T_B0_B.translation)).flatten().tolist())
        # msg = PoseStamped()
        # msg.header.stamp = rospy.Time(int(math.modf(time_ns / 1e9)[1]), int(math.modf(time_ns / 1e9)[0] * 1e9))
        # msg.pose.position.x = T_B0_B.translation[0, 0]
        # msg.pose.position.y = T_B0_B.translation[1, 0]
        # msg.pose.position.z = T_B0_B.translation[2, 0]
        # msg.pose.orientation.w = T_B0_B.rotation.w
        # msg.pose.orientation.x = T_B0_B.rotation.x
        # msg.pose.orientation.y = T_B0_B.rotation.y
        # msg.pose.orientation.z = T_B0_B.rotation.z
        # outbag.write("truth/pose", msg, msg.header.stamp)



    # Convert Truth to Proper Frame
    tlist = []
    T_I_B0 = None
    for i, row in enumerate(tqdm(vicon_file)):
        if i == 0: continue
        time_ns = int(row[0])
        if time_ns < t0: continue
        if (time_ns - t0) > duration * 1e9: break;

        t_NWU_V = np.array(row[1:4], dtype=np.float64)
        q_NWU_V = np.array(row[4:], dtype=np.float64)
        tlist.append(row[1:])
        T_NWU_V = Transform(Quaternion(np.array([q_NWU_V]).T), np.array([t_NWU_V]).T)

        T_I_B = T_I_NWU * T_NWU_V * T_V_B

        if T_I_B0 is None:
            T_I_B0 = Transform(Quaternion.from_euler(0, 0, T_I_B.rotation.euler[2,0]), T_I_B.translation)

        T_B0_B = T_I_B0.inverse * T_I_B

        tlist.append(np.vstack((T_B0_B.rotation.arr, T_B0_B.translation)).flatten().tolist())

        msg = PoseStamped()
        msg.header.stamp = rospy.Time(int(math.modf(time_ns / 1e9)[1]), int(math.modf(time_ns / 1e9)[0] * 1e9))
        msg.pose.position.x = T_B0_B.translation[0,0]
        msg.pose.position.y = T_B0_B.translation[1,0]
        msg.pose.position.z = T_B0_B.translation[2,0]
        msg.pose.orientation.w = T_B0_B.rotation.w
        msg.pose.orientation.x = T_B0_B.rotation.x
        msg.pose.orientation.y = T_B0_B.rotation.y
        msg.pose.orientation.z = T_B0_B.rotation.z
        outbag.write("truth/pose", msg, msg.header.stamp)

    # Add the Images
    for f in tqdm(sorted(listdir(images_dir))):
        if '.png' not in f: continue
        img = scipy.misc.imread(images_dir + "/" + f)
        time_ns = int(f.split('.')[0])
        if (time_ns - t0) > duration * 1e9: break;


        msg = Image()
        msg.header.frame_id = "body"
        msg.header.stamp = rospy.Time(int(math.modf(time_ns/1e9)[1]), int(math.modf(time_ns/1e9)[0]*1e9))
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = "mono8"
        msg.step = img.shape[1]
        msg.data = img.tobytes()
        outbag.write("color", msg, msg.header.stamp)

    sys.stdout.flush()
    outbag.close()


# imu = np.array(imu)
# plt.figure(1)
# plt.subplot(211)
# plt.title("gyro")
# plt.plot(imu[:,:3])
# plt.subplot(212)
# plt.title("acc")
# plt.plot(imu[:,3:])
#
# truth = np.array(truth)
# plt.figure(2)
# plt.subplot(211)
# plt.title("quat")
# plt.plot(truth[:,:4])
# plt.subplot(212)
# plt.title("t")
# plt.plot(truth[:,4:])

# plt.show()









