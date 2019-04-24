#!/usr/bin/env python

import os
import numpy as np
from data import History
from plot_helper import plot_side_by_side, init_plots, get_colors
from tqdm import tqdm
import scipy.signal
import matplotlib.pyplot as plt
from pyquat import Quaternion
import sys
from import_log import import_log

def main():
    make_feature_plots = '-f' in sys.argv
    make_input_plots = '-u' in sys.argv
    make_derivative_plots = '-d' in sys.argv
    make_performance_plot = '-p' in sys.argv

    # Shift truth timestamp
    offset = 0.0

    plot_cov = True
    pose_cov = True

    log_dir = os.path.dirname(os.path.realpath(__file__)) + "/../logs/"
    log_folders =  [int(name) for name in os.listdir(log_dir) if os.path.isdir(log_dir + name)]

    latest_folder = max(log_folders)
    fig_dir = log_dir + str(latest_folder) + "/plots"
    if not os.path.isdir(fig_dir): os.system("mkdir " + fig_dir)

    h = import_log(latest_folder)

    # Create Plots
    start = h.t.xhat[0]
    end = h.t.xhat[-1]
    init_plots(start, end, fig_dir, make_feature_plots)

    # Summarize Results
    print("\nFinal bias States")
    print("Accel", h.xhat[-1, 10:13])
    print("Gyro", h.xhat[-1, 13:16])
    print("Drag", h.xhat[-1, 16])

    # Calculate error over distance travelled
    distance_traveled = np.hstack((np.zeros(1), np.cumsum(np.sum(np.diff(h.global_pos, axis=0)**2, axis=1)**0.5)))
    error = np.sum((h.global_pos - h.global_pos_hat)**2, axis=1)**0.5
    percent_error = error/distance_traveled
    print("\n Error Over Distance Travelled: %.4f" % percent_error[-1])
    plt.figure(figsize=(14,10))
    plt.plot(percent_error[100:])
    plt.title('Error over Distance Traveled')
    plt.savefig(fig_dir +"/error_over_distance.svg", bbox_inches='tight')
    plt.close()

    # PLOT Trajectory
    plt.figure(figsize=(14, 10))
    plt.plot(h.global_pos_hat[:,0], h.global_pos_hat[:,1], label='estimated')
    plt.plot(h.global_pos[:,0], h.global_pos[:,1], label="truth")
    plt.title("trajectory")
    plt.savefig(fig_dir + "/trajectory.svg", bbox_inches="tight")
    plt.legend()
    plt.close()

    # # PLOT STATES
    plot_side_by_side(r'$p_{b/I}^I$', 0, 3, h.t.global_pos_hat, h.global_pos_hat, cov=h.cov if pose_cov else None, truth_t=h.t.global_pos, truth=h.global_pos, labels=['p_x', 'p_y', 'p_z'], start_t=start, end_t=end)
    plot_side_by_side(r'$p_{b/n}^n$', 0, 3, h.t.xhat, h.xhat, cov=h.cov if pose_cov else None, truth_t=h.t.pos, truth=h.pos, labels=['p_x', 'p_y', 'p_z'], start_t=start, end_t=end, active_arr=h.pos_active)
    plot_side_by_side(r'$v_{b/I}^b$', 3, 6, h.t.xhat, h.xhat, cov=h.cov if plot_cov else None, truth_t=h.t.vel, truth=h.vel, labels=['v_x', 'v_y', 'v_z'], start_t=start, end_t=end)
    plot_side_by_side('relative_euler', 0, 3, h.t.xhat, h.euler_hat, truth_t=h.t.att, truth=h.euler, start_t=start, end_t=end, labels=[r'\phi', r'\theta', r'\psi'])
    plot_side_by_side('global_euler', 0, 3, h.t.global_att_hat, h.global_euler_hat, truth_t=h.t.global_att, truth=h.global_euler, start_t=start, end_t=end, labels=[r'\phi', r'\theta', r'\psi'])
    # plot_side_by_side(r'$q_n^b$', 6, 10, h.t.xhat, h.xhat, cov=None, truth_t=h.t.att, truth=h.att, labels=['q_w','q_x', 'q_y', 'q_z'], start_t=start, end_t=end)
    # plot_side_by_side(r'$q_I^b$', 0, 4, h.t.global_att_hat, h.global_att_hat, cov=None, truth_t=h.t.global_att, truth=h.global_att, labels=['q_w','q_x', 'q_y', 'q_z'], start_t=start, end_t=end)
    plot_side_by_side(r'$\beta_{a}$', 10, 13, h.t.xhat, h.xhat, labels=[r'\beta_{a,x}', r'\beta_{a,y}', r'\beta_{a,z}'], start_t=start, end_t=end, cov=h.cov if plot_cov else None, cov_bounds=(9,12))
    plot_side_by_side(r'$\beta_{\omega}$', 13, 16, h.t.xhat, h.xhat, labels=[r'\beta_{\omega,x}', r'\beta_{\omega,y}', r'\beta_{\omega,z}'], start_t=start, end_t=end, cov=h.cov if plot_cov else None, cov_bounds=(12,15))
    plot_side_by_side('drag', 16, 17, h.t.xhat, h.xhat, labels=['b'], start_t=start, end_t=end, cov=h.cov if plot_cov else None, cov_bounds=(15,16))

    # PLOT INPUTS AND MEASUREMENTS
    if make_input_plots:
        b_acc, a_acc = scipy.signal.butter(6, 0.05)
        acc_smooth = scipy.signal.filtfilt(b_acc, a_acc, h.acc, axis=0)
        plot_side_by_side('$y_{a}$', 0, 2, h.t.acc, acc_smooth, truth=h.acc, truth_t=h.t.acc, labels=[r'y_{a,x}', r'y_{a,y}'], start_t=start, end_t=end)
        plot_side_by_side('u_acc', 0, 3, h.t.u_acc, h.u_acc, labels=[r'u_{a,x}', r'u_{a,y}', r'u_{a,z}'], start_t=start, end_t=end)
        plot_side_by_side('u_gyro', 0, 3, h.t.u_gyro, h.u_gyro, labels=[r'u_{\omega,x}', r'u_{\omega,y}', r'u_{\omega,z}'], start_t=start, end_t=end)
        # plot_side_by_side('$y_{alt}$', 0, 1, h.t.alt, h.alt_hat[:,None], truth=h.alt[:,None], truth_t=h.t.alt, labels=[r'-p_z'], start_t=start, end_t=end)

    # PLOT DERIVATIVES
    if make_derivative_plots:
        plot_side_by_side(r'$\dot{p}_{b/n}^n$', 0, 3, h.t.xdot, h.xdot, labels=[r'\dot{p}_x', r'\dot{p}_y', r'\dot{p}_z'], start_t=start, end_t=end)
        plot_side_by_side(r'$\dot{v}_{b/I}^b$', 3, 6, h.t.xdot, h.xdot, labels=[r'\dot{v}_x', r'\dot{v}_y', r'\dot{v}_z'], start_t=start, end_t=end)
        plot_side_by_side(r'$\dot{q}_{b/I}$', 6, 9, h.t.xdot, h.xdot, labels=[r'\dot{q}_x', r'\dot{q}_y', r'\dot{q}_z'], start_t=start, end_t=end)
        plot_side_by_side(r'$\dot{\beta}_{a}$', 9, 12, h.t.xdot, h.xdot, labels=[r'\dot{\beta}_{a,x}', r'\dot{\beta}_{a,y}', r'\dot{\beta}_{a,z}'], start_t=start, end_t=end)
        plot_side_by_side(r'$\dot{\beta}_{\omega}$', 12, 15, h.t.xdot, h.xdot, labels=[r'\dot{\beta}_{\omega,x}', r'\dot{\beta}_{\omega,y}', r'\dot{\beta}_{\omega,z}'], start_t=start, end_t=end)
        plot_side_by_side(r'$\dot{b}$', 15, 16, h.t.xdot, h.xdot, labels=[r'\dot{b}'], start_t=start, end_t=end)

    # PLOT Performance Results
    if make_performance_plot:
        plt.figure(figsize=(14,10))
        plt.hist(h.prop_time[h.prop_time > 0.001], bins=35, alpha=0.5, label='propagation')
        plt.hist(h.feat_time[h.feat_time > 0], bins=5, alpha=0.5, label='feature update')
        plt.legend()
        plt.savefig(fig_dir+"/perf.svg", bbox_inches='tight')
        plt.close()

    if make_feature_plots:
        plt.figure(figsize=(16, 10))
        colors = get_colors(35, plt.cm.jet)
        for i in h.ids:
            plt.subplot(211)
            plt.plot(h.t.feat_hat[i], h.feat_hat[i][:,0], color=colors[int(i%35)])
            plt.subplot(212)
            plt.plot(h.t.feat_hat[i], h.feat_hat[i][:,1], color=colors[int(i%35)])
        plt.savefig(fig_dir + "/features.svg", bbox_inches='tight')

        plt.figure(figsize=(16, 10))
        plt.title('depth')
        for i in h.ids:
            plt.plot(h.t.depth_hat[i], h.depth_hat[i][:], color=colors[int(i%35)])
        plt.savefig(fig_dir + "/depth.svg", bbox_inches='tight')

        for i in tqdm(h.ids):
            if i not in h.feat_hat:
                continue
            plot_side_by_side('x_{}'.format(i), 0, 2, h.t.feat_hat[i], h.feat_hat[i], truth_t=h.t.feat[i],
                              truth=h.feat[i], labels=['u', 'v'], start_t=start, end_t=end, subdir='lambda',
                              cov=h.feat_cov[i] if plot_cov else None)
            if hasattr(h, 'depth_hat'):
                plot_side_by_side('x_{}'.format(i), 0, 1, h.t.depth_hat[i], h.depth_hat[i][:, None], truth_t=h.t.depth[i] if hasattr(h, 'depth') else None,
                              truth=h.depth[i][:, None] if hasattr(h, 'depth') else None, labels=[r'\frac{1}{\rho}'], start_t=start, end_t=end,
                              cov=h.depth_cov[i] if plot_cov else None, subdir='rho')



if __name__ == '__main__':
    main()






