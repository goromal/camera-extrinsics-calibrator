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

# Rotation from NWU to NED
R_NWU_NED = np.array([[1., 0, 0],
                      [0, -1., 0],
                      [0, 0, -1.]])
q_NWU_NED = Quaternion.from_R(R_NWU_NED)

R_DWN_NED= np.array([[0., 0., 1.],
                     [0., -1., 0.],
                     [1., 0., 0.]])
q_DWN_NED=Quaternion.from_R(R_DWN_NED)


# Ground Truth is the state of the IMU frame with respect to the world frame
# It appears that the world frame is in DWN, I figure that from inspection of the pose trajectory
# The IMU frame is different from the body frame of the vehicle

# Rotation from the IMU frame to the body frame
R_IMU_B = np.array([[ 0.33638, -0.01749,  0.94156],
                    [ 0.02078,  0.99972,  0.01114],
                    [-0.9415 ,  0.01582,  0.33665]])
q_IMU_B = Quaternion.from_R(R_IMU_B)

R_IMU_NED = R_DWN_NED.dot(R_IMU_B.T)

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

def load_data(folder, start=0, end=np.inf, sim_features=False, show_image=False):
    # First, load IMU data
    csvfile = open(folder+'/imu0/data.csv', 'rb')
    imu_data = []
    reader = csv.reader(csvfile)
    for i, row in tqdm(enumerate(reader)):
        if i > 0:
            imu_data.append([float(item) for item in row])
    imu_data = np.array(imu_data)

    # rotate IMU data into the NED frame
    imu_data[:,1:4] = q_IMU_B.R.dot(imu_data[:,1:4].T).T
    imu_data[:,4:7] = q_IMU_B.R.dot(imu_data[:,4:7].T).T

    # Adjust time stamp so it's reasonable
    t0 = imu_data[0,0]
    imu_data[:,0] -= t0
    imu_data[:,0] /= 1e9

    plt.figure(1)
    plt.plot(imu_data[:,4:7])

    # Load ground truth estimate
    ground_truth = []
    csvfile = open(folder + '/state_groundtruth_estimate0/data.csv', 'rb')
    reader = csv.reader(csvfile)
    for i, row in tqdm(enumerate(reader)):
        if i > 0:
            ground_truth.append([float(item) for item in row])
    ground_truth = np.array(ground_truth)
    ground_truth[:, 0] -= t0
    ground_truth[:, 0] /= 1e9

    # Rotate ground truth into correct frame (it originally comes in in the body frame)
    ground_truth[:, 1:4] = R_IMU_NED.dot(ground_truth[:,1:4].T).T
    ground_truth[:, 1:4] = R_IMU_NED.dot(ground_truth[:, 1:4].T).T

    plt.figure(2)
    ax = plt.subplot(111, projection='3d')
    plt.plot(ground_truth[:1, 1],
             ground_truth[:1, 2],
             ground_truth[:1, 3], 'kx')
    plt.plot(ground_truth[:4000, 1],
             ground_truth[:4000, 2],
             ground_truth[:4000, 3])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

    if sim_features:
        # Simulate Landmark Measurements
        landmarks = np.random.uniform(-25, 25, (2,3))
        feat_time, zetas, depths, ids = add_landmark(ground_truth, landmarks)

    else:
        # Load Camera Data and calculate Landmark Measurements
        images0 = []
        images1 = []
        image_time = []
        csvfile = open(folder + '/cam0/data.csv', 'rb')
        reader = csv.reader(csvfile)
        for i, row in tqdm(enumerate(reader)):
            if i > 0:
                image_time.append((float(row[0]) - t0) / 1e9)
                images0.append(folder + '/cam0/data/' + row[1])
                images1.append(folder + '/cam1/data/' + row[1])
                if show_image:
                    cv2.imshow('image', cv2.imread(folder+'/cam0/data/' + row[1]))
                    print image_time[-1]
                    #cv2.waitKey(0)
        image_time = np.array(image_time)
        # images = np.array(images)

        with open(folder + '/cam0/sensor.yaml', 'r') as stream:
            try:
                data = yaml.load(stream)

                cam0_sensor = {
                    'resolution': np.array(data['resolution']),
                    'intrinsics': np.array(data['intrinsics']),
                    'rate_hz': data['rate_hz'],
                    'distortion_coefficients': np.array(data['distortion_coefficients']),
                    'body_to_sensor_transform': np.array(data['T_BS']['data']).reshape(
                        (data['T_BS']['rows'], data['T_BS']['cols']))
                }


            except yaml.YAMLError as exc:
                print(exc)

        with open(folder + '/cam1/sensor.yaml', 'r') as stream:
            try:
                data = yaml.load(stream)
                cam1_sensor = {
                    'resolution': data['resolution'],
                    'intrinsics': data['intrinsics'],
                    'rate_hz': data['rate_hz'],
                    'distortion_coefficients': np.array(data['distortion_coefficients']),
                    'body_to_sensor_transform': np.array(data['T_BS']['data']).reshape(
                        (data['T_BS']['rows'], data['T_BS']['cols']))
                }

            except yaml.YAMLError as exc:
                print(exc)

    # chop data
    imu_data = imu_data[(imu_data[:,0] > start) & (imu_data[:,0] < end), :]
    if sim_features:
        for l in range(len(landmarks)):
            zetas[l] = zetas[l][(feat_time > start) & (feat_time < end)]
            depths[l] = depths[l][(feat_time > start) & (feat_time < end)]
        ids = ids[(feat_time > start) & (feat_time < end)]
    else:
        images0 = [f for f, t in zip(images0, (image_time > start) & (image_time < end)) if t]
        images1 = [f for f, t in zip(images1, (image_time > start) & (image_time < end)) if t]
        image_time = image_time[(image_time > start) & (image_time < end)]
    ground_truth = ground_truth[(ground_truth[:, 0] > start) & (ground_truth[:, 0] < end), :]

    out_dict = dict()
    out_dict['imu'] = imu_data
    out_dict['truth'] = ground_truth
    if sim_features:
        out_dict['feat_time'] = feat_time
        out_dict['zetas'] = zetas
        out_dict['depths'] = depths
        out_dict['ids'] = ids
    else:
        out_dict['cam0_sensor'] = cam0_sensor
        out_dict['cam1_sensor'] = cam1_sensor
        out_dict['cam0_frame_filenames'] = images0
        out_dict['cam1_frame_filenames'] = images1
        out_dict['cam_time'] = image_time

    return out_dict



if __name__ == '__main__':
    data = load_data('/mnt/pccfs/not_backed_up/eurocmav/mav0', show_image=True)
    save_to_file('/mnt/pccfs/not_backed_up/eurocmav/mav0/data.npy', data)
    print "done"
