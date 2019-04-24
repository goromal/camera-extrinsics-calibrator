import os
import numpy as np
from data import History
from plot_helper import plot_side_by_side, init_plots, get_colors
from tqdm import tqdm
import scipy.signal
import matplotlib.pyplot as plt
from pyquat import Quaternion
from import_log import import_log
import sys

bag_file = 'V1_03_difficult'
# bag_file = 'V1_02_medium'
fig_dir = os.path.dirname(os.path.realpath(__file__)) + "/../results/" + bag_file + "/"


def plot_paper():
    data = {}

    global bag_file

    # data['V1_01_easy'] = {}
    # data['V1_01_easy']['KR'] = import_log(1527114159)
    # data['V1_01_easy']['B'] = import_log(1527114087)
    # data['V1_01_easy']['PU'] = import_log(1527114239)
    # data['V1_01_easy']['DT'] = import_log(1527114382)
    # data['V1_01_easy']['PU+DT+KF'] = import_log(1527114602)

    # V1_03_difficult
    data['V1_03_difficult'] = {}
    data['V1_03_difficult']['B'] = import_log(1527189096)
    data['V1_03_difficult']['DT'] = import_log(1527189307)
    data['V1_03_difficult']['PU'] = import_log(1527189202)
    data['V1_03_difficult']['KR'] = import_log(1527189149)
    data['V1_03_difficult']['PU+DT+KR'] = import_log(1527189489)

    # V1_02_medium
    # data['V1_02_medium'] = {}
    # data['V1_02_medium']['PU'] = import_log(1527095547)
    # data['V1_02_medium']['DT'] = import_log(1527096631)
    # data['V1_02_medium']['PU+DT'] = import_log(1527096534)
    # data['V1_02_medium']['B'] = import_log(1526506904)
    # data['V1_02_medium']['PU+DT+KF'] = import_log(1527091775)


    ## Old data
    # data['PU'] = import_log(1526503296)
    # data['PU+DT'] = import_log(1526503405)
    # data['B'] = import_log(1526506904)
    # data['PU+DT+KF'] = import_log(1526509777)

    global fig_dir
    if not os.path.isdir(fig_dir): os.system("mkdir -p " + fig_dir)


    plot_velocities(data[bag_file])
    plot_attitude(data[bag_file])
    plot_positions(data[bag_file])
    plot_biases(data[bag_file])
    plot_zoomed(data[bag_file])
    plt.show()

def plot_zoomed(data):
    global fig_dir, bag_file

    colors = get_colors(len(data)+1, plt.cm.jet)

    plt.figure(figsize=(16,10))
    ax = None

    # Plot x velocity
    ax=plt.subplot(4,1,1, sharex=ax)
    plt.plot(data['B'].t.vel, data['B'].vel[:,0], '--', label="truth", color=colors[0], linewidth=2)
    for j, (key, d) in enumerate(data.iteritems()):
        plt.plot(d.t.xhat, d.xhat[:, 3], lines[j], label=key, color=colors[j+1], linewidth=2)
    plt.ylabel(r"$v_x$ (m/s)")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=False, shadow=False, ncol=6)

    # Plot pitch
    ax=plt.subplot(4,1,2, sharex=ax)
    plt.plot(data['B'].t.att, data['B'].euler[:,1], '--', label="truth", color=colors[0], linewidth=2)
    for j, (key, d) in enumerate(data.iteritems()):
        plt.plot(d.t.global_att, d.global_euler_hat[:,2], lines[j], label=key, color=colors[j+1], linewidth=2)
    plt.ylabel(r"pitch (rad)")

    # Plot Y gyro bias
    ax=plt.subplot(4,1,3, sharex=ax)
    for j, (key, d) in enumerate(data.iteritems()):
        plt.plot(d.t.xhat, d.xhat[:,11], lines[j], label=key, color=colors[j+1], linewidth=2)
    plt.ylabel(r"pitch (rad)")

    # Plot X accel bias
    ax=plt.subplot(4,1,4, sharex=ax)
    for j, (key, d) in enumerate(data.iteritems()):
        plt.plot(d.t.xhat, d.xhat[:,14], lines[j], label=key, color=colors[j+1], linewidth=2)
    plt.ylabel(r"pitch (rad)")


def plot_velocities(data):
    global fig_dir, bag_file

    colors = get_colors(len(data)+1, plt.cm.jet)

    plt.figure(figsize=(16,10))
    ax = None
    lines = ['-' for i in range(len(data))]
    for i in range(3):
        ax = plt.subplot(3, 1, i + 1, sharex=ax)
        plt.plot(data['B'].t.vel, data['B'].vel[:, i], '--', label="truth", color=colors[0], linewidth=2)
        for j, (key, d) in enumerate(data.iteritems()):
            plt.plot(d.t.xhat, d.xhat[:, 3+i], lines[j], label=key, color=colors[j+1], linewidth=2)
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=False, shadow=False, ncol=6)
        plt.ylabel("m/s")
    plt.xlabel("s")
    plt.savefig(fig_dir + "velocities.pdf", bbox_inches='tight', dpi=600)


def plot_attitude(data):
    global fig_dir

    colors = get_colors(len(data)+1, plt.cm.jet)

    plt.figure(figsize=(16,10))
    ax = None
    titles = ['roll', 'pitch', 'yaw']
    lines = ['-' for i in range(len(data))]
    for i in range(3):
        ax = plt.subplot(3,1,i+1, sharex=ax)
        plt.plot(data['B'].t.att, data['B'].euler[:, i], '--', label="truth", color=colors[0], linewidth=2)
        for j, (key, d) in enumerate(data.iteritems()):
            plt.plot(d.t.global_att, d.global_euler_hat[:,i], lines[j], label=key, color=colors[j+1], linewidth=2)
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=False, shadow=False, ncol=6)
        plt.ylabel(titles[i] + ' (rad)')
    plt.xlabel("s")
    plt.savefig(fig_dir + "attitude.pdf", bbox_inches='tight', dpi=600)

def plot_positions(data):
    global fig_dir, bag_file

    colors = get_colors(len(data)+1, plt.cm.jet)

    plt.figure(figsize=(16,10))
    ax = None
    lines = ['-' for i in range(len(data))]
    for i in range(3):
        ax = plt.subplot(3, 1, i + 1, sharex=ax)
        plt.plot(data['B'].t.pos, data['B'].pos[:, i], '--', label="truth", color=colors[0], linewidth=2)
        for j, (key, d) in enumerate(data.iteritems()):
            plt.plot(d.t.global_pos_hat, d.global_pos_hat[:, i], lines[j], label=key, color=colors[j+1], linewidth=2)
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=False, shadow=False, ncol=6)
        plt.ylabel("m")
    plt.xlabel("s")
    plt.savefig(fig_dir + "positions.pdf", bbox_inches='tight', dpi=600)

def plot_biases(data):
    global fig_dir, bag_file
    colors = get_colors(len(data)+1, plt.cm.jet)
    plt.figure(figsize=(16,10))
    ax = None
    lines = ['-' for i in range(len(data))]

    for i in range(7):
        ax = plt.subplot(7, 1, i+1, sharex=ax)
        for j, (key, d) in enumerate(data.iteritems()):
            plt.plot(d.t.xhat, d.xhat[:,10+i], lines[j], label=key, color=colors[j+1], linewidth=2)
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=False, shadow=False, ncol=6)
        if i < 3:
            plt.xlabel(r"$\beta_a$ (m/s^2)")
        elif i < 6:
            plt.xlabel(r"$\beta_g$ (rad/s)")
        else:
            plt.xlabel("b (1/s)")
    plt.xlabel("s")
    plt.savefig(fig_dir + "biases.pdf", bbox_inches='tight', dpi=600)



if __name__ == '__main__':
    plot_paper()




