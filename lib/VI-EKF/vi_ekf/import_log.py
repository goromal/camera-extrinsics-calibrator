import os
import numpy as np
from data import History
from plot_helper import plot_side_by_side, init_plots, get_colors
from tqdm import tqdm
import scipy.signal
import matplotlib.pyplot as plt
from pyquat import Quaternion
import sys

def import_log(log_stamp):
    log_dir = os.path.dirname(os.path.realpath(__file__)) + "/../logs/"
    prop_file = open(log_dir + str(log_stamp) + "/prop.txt")
    perf_file = open(log_dir + str(log_stamp) + "/perf.txt")
    meas_file = open(log_dir + str(log_stamp) + "/meas.txt")
    conf_file = open(log_dir + str(log_stamp) + "/conf.txt")
    input_file = open(log_dir + str(log_stamp) + "/input.txt")
    xdot_file = open(log_dir + str(log_stamp) + "/xdot.txt")

    h = History()
    len_prop_file = 0
    for line in prop_file:
        line_arr = np.array([float(item) for item in line.split()])
        if len_prop_file == 0: len_prop_file = len(line_arr)
        if len(line_arr) < len_prop_file: continue
        num_features = (len(line_arr) - 34) / 8
        X = 1
        COV = 1 + 17 + 5*num_features
        t = line_arr[0]
        h.store(t, xhat=line_arr[1:18], cov=np.diag(line_arr[COV:]), num_features=num_features)

    for i, line in enumerate(perf_file):
        if i == 0: continue
        line_arr = np.array([float(item) for item in line.split()])
        if len(line_arr) == 12:
            t = line_arr[0]
            h.store(t, prop_time=line_arr[1], acc_time=line_arr[2], pos_time=line_arr[5], feat_time=line_arr[8], depth_time=line_arr[10])


    ids = []
    for line in meas_file:
        try:
            meas_type = line.split()[0]
            line_arr = np.array([float(item) for item in line.split()[2:]])
            t = float(line.split()[1])
        except:
            continue


        if meas_type == 'ACC':
            if len(line_arr) < 6: continue
            h.store(t, acc=line_arr[0:2], acc_hat=line_arr[2:4], acc_active=line_arr[5])
        elif meas_type == 'ATT':
            if len(line_arr) < 10: continue
            h.store(t, att=line_arr[0:4], att_hat=line_arr[4:8], att_active=line_arr[9])
        elif meas_type == 'GLOBAL_ATT':
            if len(line_arr) < 8: continue
            h.store(t, global_att=line_arr[0:4], global_att_hat=line_arr[4:8])
        elif meas_type == 'POS':
            if len(line_arr) < 8: continue
            h.store(t, pos=line_arr[0:3], pos_hat=line_arr[3:6], pos_active=line_arr[7])
        elif meas_type == 'GLOBAL_POS':
            if len(line_arr) < 6: continue
            h.store(t, global_pos=line_arr[0:3], global_pos_hat=line_arr[3:6])
        elif meas_type == 'FEAT':
            if len(line_arr) < 7: continue
            id = line_arr[6]
            h.store(t, id, feat_hat=line_arr[0:2], feat=line_arr[2:4], feat_cov=np.diag(line_arr[4:6]))
            ids.append(id) if id not in ids else None
        elif meas_type == 'DEPTH':
            if len(line_arr) < 4: continue
            # Invert the covariance measurement
            p = 1.0/line_arr[0]
            s = line_arr[2]
            cov = 1./(p+s) - 1./p
            h.store(t, line_arr[3], depth=line_arr[1], depth_hat=line_arr[0], depth_cov=[[cov   ]])
        elif meas_type == 'ALT':
            if len(line_arr) < 3: continue
            h.store(t, alt=line_arr[0], alt_hat=line_arr[1])
        else:
            print("unsupported measurement type ", meas_type)

    for line in input_file:
        line_arr = np.array([float(item) for item in line.split()])
        if len(line_arr) < 6: continue
        h.store(line_arr[0], u_acc=line_arr[1:4], u_gyro=line_arr[4:])

    for line in xdot_file:
        line_arr = np.array([float(item) for item in line.split()])
        if len(line_arr) < 18: continue
        h.store(line_arr[0], dt=line_arr[1], xdot=line_arr[2:18])

    h.tonumpy()

    # Calculate true body-fixed velocity by differentiating position and rotating
    # into the body frame
    delta_t = np.diff(h.t.global_pos)
    good_ids = delta_t > 0.001 # only take truth measurements with a reasonable time difference
    delta_t = delta_t[good_ids]
    h.t.vel = h.t.global_pos[np.hstack((good_ids, False))]
    delta_x = np.diff(h.global_pos, axis=0)
    delta_x = delta_x[good_ids]
    unfiltered_inertial_velocity = np.vstack(delta_x / delta_t[:, None]) # Differentiate Truth
    b_vel, a_vel = scipy.signal.butter(3, 0.50) # Smooth Differentiated Truth
    v_inertial = scipy.signal.filtfilt(b_vel, a_vel, unfiltered_inertial_velocity, axis=0)

    # Rotate into Body Frame
    vel_data = []
    try:
        att = h.global_att[np.hstack((good_ids))]
    except:
        att = h.global_att[np.hstack((good_ids, False))]
    for i in range(len(h.t.vel)):
        q_I_b = Quaternion(att[i, :, None])
        vel_data.append(q_I_b.invrot(v_inertial[i, None].T).T)

    h.vel = np.array(vel_data).squeeze()

    # Calculate Euler Angles from attitudes
    # Convert global attitude to euler angles
    true_euler, est_euler = np.zeros((len(h.global_att),3)), np.zeros((len(h.global_att_hat),3))
    for i, true_quat in enumerate(h.global_att): true_euler[i,:,None] = Quaternion(true_quat[:,None]).euler
    for i, est_quat in enumerate(h.global_att_hat): est_euler[i,:,None] = (Quaternion(est_quat[:,None]).euler)
    h.global_euler = true_euler
    h.global_euler_hat = est_euler

    # Convert relative attitude to euler angles
    true_euler, est_euler = np.zeros((len(h.att),3)), np.zeros((len(h.xhat),3))
    for i, true_quat in enumerate(h.att): true_euler[i,:,None] = Quaternion(true_quat[:,None]).euler
    for i, est_quat in enumerate(h.xhat[:,6:10]): est_euler[i,:,None] = (Quaternion(est_quat[:,None]).euler)
    h.euler = true_euler
    h.euler_hat = est_euler

    h.ids = ids

    return h