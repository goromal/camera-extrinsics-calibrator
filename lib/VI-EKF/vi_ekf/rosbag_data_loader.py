import csv
import numpy as np
import glob, os, sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from tqdm import tqdm
from pyquat import Quaternion
from add_landmark import add_landmark
import yaml
import matplotlib.pyplot as plt
import rosbag
import rospy
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
from plot_helper import plot_3d_trajectory, play_trajectory_with_sim_features
import klt_tracker
import struct
import scipy.interpolate
import cPickle
from math_helper import norm, e_z

def to_list(vector3):
    return [vector3.x, vector3.y, vector3.z]

def to_list4(quat):
    return [quat.w, quat.x, quat.y, quat.z]

def load_from_file(filename):
    data = np.load(filename)
    return data.item()

def save_to_file(filename, data):
    np.save(filename, data)

def make_undistort_funtion(intrinsics, resolution, distortion_coefficients):
    A = np.array([[float(intrinsics[0]), 0., float(intrinsics[2])], [0., float(intrinsics[1]), float(intrinsics[3])], [0., 0., 1.]])
    Ap, _ = cv2.getOptimalNewCameraMatrix(A, distortion_coefficients, (resolution[0], resolution[1]), 1.0)

    def undistort(image):
        return cv2.undistort(image, A, distortion_coefficients, None, Ap)

    return undistort, Ap

def calculate_velocity_from_position(t, position, orientation):
    # Calculate body-fixed velocity by differentiating position and rotating
    # into the body frame
    b, a = scipy.signal.butter(8, 0.03)  # Create a Butterworth Filter
    # differentiate Position
    delta_x = np.diff(position, axis=0)
    delta_t = np.diff(t)
    unfiltered_inertial_velocity = np.vstack((np.zeros((1, 3)), delta_x / delta_t[:, None]))
    # Filter
    v_inertial = scipy.signal.filtfilt(b, a, unfiltered_inertial_velocity, axis=0)
    # Rotate into Body Frame
    vel_data = []
    for i in range(len(t)):
        q_I_b = Quaternion(orientation[i, :, None])
        vel_data.append(q_I_b.rot(v_inertial[i, None].T).T)

    vel_data = np.array(vel_data).squeeze()
    return vel_data

# https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def load_data(filename, start=0, end=np.inf, sim_features=False, show_image=False, plot_trajectory=True):
    cam_center = np.array([[316.680559, 230.660661]]).T
    cam_F = np.array([[533.013144, 0.0, 0.0], [0.0, 533.503964, 0.0]])

    if not os.path.isfile('data/bag_tmp.pkl'):
        print "loading rosbag", filename
        # First, load IMU data
        bag = rosbag.Bag(filename)
        imu_data = []
        truth_pose_data = []
        image_data = []
        depth_data = []
        image_time = []
        depth_time = []

        topic_list = ['/imu/data',
                      '/vrpn_client_node/Leo/pose',
                      '/vrpn/Leo/pose',
                      '/vrpn/RGBD_Camera/pose',
                      '/baro/data',
                      '/sonar/data',
                      '/is_flying',
                      '/gps/data',
                      '/mag/data',
                      '/camera/depth/image_rect',
                      '/camera/rgb/image_rect_mono/compressed',
                      '/camera/rgb/image_rect_color/compressed']

        for topic, msg, t in tqdm(bag.read_messages(topics=topic_list), total=bag.get_message_count(topic_list)):
            if topic == '/imu/data':
                imu_meas = [msg.header.stamp.to_sec(),
                            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                imu_data.append(imu_meas)

            if topic == '/vrpn_client_node/Leo/pose' or topic == '/vrpn/Leo/pose' or topic == '/vrpn/RGBD_Camera/pose':
                truth_meas = [msg.header.stamp.to_sec(),
                              msg.pose.position.z, -msg.pose.position.x, -msg.pose.position.y,
                              -msg.pose.orientation.w, -msg.pose.orientation.z, msg.pose.orientation.x, msg.pose.orientation.y]
                truth_pose_data.append(truth_meas)

            if topic == '/camera/rgb/image_rect_mono/compressed' or topic == '/camera/rgb/image_rect_color/compressed':
                np_arr = np.fromstring(msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                image_time.append(msg.header.stamp.to_sec())
                image_data.append(image)



            if topic == '/camera/depth/image_rect':
                img = np.fromstring(msg.data, np.float32).reshape(msg.height, msg.width)

                depth_time.append(msg.header.stamp.to_sec())
                depth_data.append(img)

        tmp_dict = dict()
        tmp_dict['imu_data'] = np.array(imu_data)
        tmp_dict['truth_pose_data'] = np.array(truth_pose_data)
        tmp_dict['image_data'] = np.array(image_data)
        tmp_dict['depth_data'] = np.array(depth_data)
        tmp_dict['image_time'] = np.array(image_time)
        tmp_dict['depth_time'] = np.array(depth_time)
        cPickle.dump(tmp_dict, open('data/bag_tmp.pkl', 'wb'), protocol=2)
    else:
        tmp_dict = cPickle.load(open('data/bag_tmp.pkl', 'rb'))

    imu_data = tmp_dict['imu_data']
    truth_pose_data = tmp_dict['truth_pose_data']
    image_data = tmp_dict['image_data']
    depth_data = tmp_dict['depth_data']
    image_time = tmp_dict['image_time']
    depth_time = tmp_dict['depth_time']


    # assert np.abs(truth_pose_data[0, 0] - imu_data[0, 0]) < 1e5, 'truth and imu timestamps are vastly different: {} (truth) vs. {} (imu)'.format(truth_pose_data[0, 0], imu_data[0, 0])

    # Remove Bad Truth Measurements
    good_indexes = np.hstack((True, np.diff(truth_pose_data[:,0]) > 1e-3))
    truth_pose_data = truth_pose_data[good_indexes]

    vel_data = calculate_velocity_from_position(truth_pose_data[:,0], truth_pose_data[:,1:4], truth_pose_data[:,4:8])

    ground_truth = np.hstack((truth_pose_data, vel_data))

    # Adjust timestamp
    imu_t0 = imu_data[0,0] +1
    gt_t0 = ground_truth[0,0]
    imu_data[:,0] -= imu_t0
    ground_truth[:,0] -= gt_t0
    image_time -= imu_t0
    depth_time -= imu_t0

    # Chop Data to start and end
    imu_data = imu_data[(imu_data[:,0] > start) & (imu_data[:,0] < end), :]
    ground_truth = ground_truth[(ground_truth[:, 0] > start) & (ground_truth[:, 0] < end), :]
    image_data = image_data[(image_time > start) & (image_time < end)]
    depth_data = depth_data[(depth_time > start) & (depth_time < end)]
    image_time = image_time[(image_time > start) & (image_time < end)]
    depth_time = depth_time[(depth_time > start) & (depth_time < end)]

    # Simulate camera-to-body transform
    q_b_c = Quaternion.from_R(np.array([[0, 1, 0],
                                        [0, 0, 1],
                                        [1, 0, 0]]))
    # q_b_c = Quaternion.Identity()
    # p_b_c = np.array([[0.16, -0.05, 0.1]]).T
    p_b_c = np.zeros((3,1))

    if sim_features:
        # Simulate Landmark Measurements
        # landmarks = np.random.uniform(-25, 25, (2,3))
        # landmarks = np.array([[1, 0, 1, 0, np.inf],
        #                       [0, 1, 1, 0, np.inf],
        #                       [0, 0, 1, 0, 3],
        #                       [1, 1, 1, 0, 3],
        #                       [-1, 0, 1, 10, 25],
        #                       [0, -1, 1, 10, 25],
        #                       [-1, -1, 1, 10, np.inf],
        #                       [1, -1, 1, 20, np.inf],
        #                       [-1, 1, 1, 20, np.inf]])
        N = 500
        last_t = imu_data[-1,0]
        landmarks = np.hstack([np.random.uniform(-4, 4, (N, 1)),
                               np.random.uniform(-4, 4, (N, 1)),
                               np.random.uniform(-4, 4, (N, 1))])

        feat_time, ids, lambdas, depths, q_zetas = add_landmark(ground_truth, landmarks, p_b_c, q_b_c, cam_F, cam_center)
        play_trajectory_with_sim_features(image_time, image_data, feat_time, ids, lambdas)


    else:
        tracker = klt_tracker.KLT_tracker(3, show_image=show_image)
        lambdas, depths, ids, feat_time, q_zetas = [], [], [], [], []

        _, image_height, image_width = image_data.shape
        _, depth_height, depth_width = depth_data.shape

        assert image_height == depth_height and image_width == depth_width

        for i, image in enumerate(image_data):
            frame_lambdas, frame_ids = tracker.load_image(image, image_time[i])

            # KLT tracker doesn't consider image bounds :(
            frame_lambdas = np.clip(frame_lambdas[:, 0], [0, 0], [639, 479] )

            frame_depths = []
            frame_qzetas = []
            nearest_depth = np.abs(depth_time - image_time[i]).argmin()
            for x, y in frame_lambdas:
                d = depth_data[nearest_depth][int(y), int(x)]
                frame_depths.append(d)
                zeta = np.array([[x - cam_center[0,0], (y-cam_center[1,0])*(cam_F[1,1]/cam_F[0,0]), cam_F[0,0]]]).T
                zeta /= norm(zeta)
                frame_qzetas.append(Quaternion.from_two_unit_vectors(e_z, zeta).elements[:,0])

            depths.append(np.array(frame_depths)[:,None])
            lambdas.append(frame_lambdas)
            ids.append(frame_ids)
            feat_time.append(image_time[i])
            q_zetas.append(np.array(frame_qzetas))
        tracker.close()

    # self.undistort, P = data_loader.make_undistort_funtion(intrinsics=self.data['cam0_sensor']['intrinsics'],
    #                                                     resolution=self.data['cam0_sensor']['resolution'],
    #                                                     distortion_coefficients=self.data['cam0_sensor']['distortion_coefficients'])

    # self.inverse_projection = np.linalg.inv(P)

    # lambdas, ids = self.tracker.load_image(image)
    #
    # if lambdas is not None and len(lambdas) > 0:
    #     lambdas = np.pad(lambdas[:, 0], [(0, 0), (0, 1)], 'constant', constant_values=0)

    if plot_trajectory:
        plot_3d_trajectory(ground_truth, feat_time=feat_time, qzetas=q_zetas, depths=depths, ids=ids, p_b_c=p_b_c, q_b_c=q_b_c)

    plt.ioff()
    out_dict = dict()
    out_dict['imu'] = imu_data
    out_dict['truth'] = ground_truth
    out_dict['feat_time'] = feat_time
    out_dict['lambdas'] = lambdas
    out_dict['depths'] = depths
    out_dict['ids'] = ids
    out_dict['p_b_c'] = p_b_c
    out_dict['q_b_c'] = q_b_c
    out_dict['q_zetas'] = q_zetas
    # out_dict['image'] = image_data
    out_dict['image_t'] = image_time

    # out_dict['depth'] = depth_data
    out_dict['depth_t'] = depth_time
    out_dict['cam_center'] = cam_center
    out_dict['cam_F'] = cam_F

    return out_dict



if __name__ == '__main__':
    data = load_data('data/truth_imu_depth_mono.bag')
    print "done"

