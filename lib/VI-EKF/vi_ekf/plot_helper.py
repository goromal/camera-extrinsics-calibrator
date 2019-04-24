import matplotlib.pyplot as plt
from pyquat import Quaternion
from cycler import cycler
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import os
import cv2

plot_start = None
plot_end = None
figure_directory = None

def init_plots(start, end, fig_directory="plots/", feature_plots=False):
    # Set the colormap to 'jet'
    plt.jet()
    plt.set_cmap('jet')
    plt.rcParams['image.cmap'] = 'jet'
    plt.rcParams['figure.max_open_warning'] = 100
    if fig_directory:
        global figure_directory
        figure_directory = fig_directory

    os.system("rm -f " + fig_directory + r"/" + r"*.svg")
    if feature_plots:
        if not os.path.isdir(fig_directory + r"/lambda"):
            os.system("mkdir " + fig_directory + r"/lambda")
        if not os.path.isdir(fig_directory + r"/rho"):
            os.system("mkdir " + fig_directory + r"/rho")
        os.system("rm -f " + fig_directory + r"/lambda/*.svg")
        os.system("rm -f " + fig_directory + r"/rho/*.svg")

    global plot_start, plot_end
    plot_start = start
    plot_end = end

def get_colors(num_rows, cmap):
    return cmap(np.linspace(0, 1, num_rows))

def plot_3d_side_by_side(h, p_b_c, q_b_c):
    ground_truth = np.hstack([h.t.pos[:, None], h.pos[:, :, 0], h.att[:, :, 0], h.vel[:, :, 0]])
    estimate = np.hstack([h.t.x_hat[:, None], h.x_hat[:, :, 0]])
    qzetas, qzetahats, depths, depthhats = [], [], [], []
    for frame, frame_id in enumerate(h.ids):
        try:
            qzetas.append(np.array([h.qzeta[i][frame][:,0] for i in frame_id]))
            qzetahats.append(np.array([h.qzeta_hat[i][frame][:, 0] for i in frame_id]))
            depths.append(np.array([h.depth[i][frame][0, 0] for i in frame_id]))
            depthhats.append(np.array([h.depth_hat[i][frame][0, 0] for i in frame_id]))
        except:
            pass

    plt.figure(100, figsize=(14,10))
    ax1 = plt.subplot(211, projection='3d')
    plot_3d_trajectory(ground_truth, h.t.ids, qzetas, depths, h.ids, p_b_c, q_b_c, ax1)
    ax2 = plt.subplot(212, projection='3d')
    plot_3d_trajectory(estimate, h.t.ids, qzetahats, depthhats, h.ids, p_b_c, q_b_c, ax2)
    plt.show()


def plot_side_by_side(title, start, end, est_t, estimate, cov=None, truth_t=None, truth=None, labels=None, skip=1, 
    save=True, cov_bounds=None, start_t=None, end_t=None, truth_offset=None, subdir=None, ylim=None, show=False,
    active_arr=None):
    estimate = estimate[:, start:end]
    if cov_bounds == None:
        cov_bounds = (start,end)

    if isinstance(cov, np.ndarray):
        cov_copy = cov[:, cov_bounds[0]:cov_bounds[1], cov_bounds[0]:cov_bounds[1]].copy()

    if start_t is None:
        start_t = est_t[0]

    if end_t is None:
        end_t = est_t[-1]

    if isinstance(truth_t, np.ndarray):
        truth_t_copy = truth_t[(truth_t > start_t) & (truth_t < end_t)].copy()
        if truth_offset is not None:
            truth_t_copy -= truth_offset

    if isinstance(truth, np.ndarray):
        truth_copy = truth[(truth_t > start_t) & (truth_t < end_t)].copy()
        if isinstance(active_arr, np.ndarray):
            active_arr_copy = active_arr[(truth_t > start_t) & (truth_t < end_t)].copy()

    plt.figure(figsize=(16, 10))
    colors = get_colors(2, plt.cm.jet)

    ax = None
    for i in range(end - start):
        ax = plt.subplot(end-start, 1, i + 1, sharex=(ax if ax is not None else None))
        if isinstance(truth, np.ndarray):
            if isinstance(active_arr, np.ndarray):
                truth_t_copy_active = truth_t_copy[active_arr_copy ==1]
                truth_t_copy_inactive = truth_t_copy[active_arr_copy ==0]
                truth_copy_active = truth_copy[active_arr_copy ==1]
                truth_copy_inactive = truth_copy[active_arr_copy ==0]
                plt.plot(truth_t_copy_active[::skip], truth_copy_active[::skip, i], color='g')
                plt.plot(truth_t_copy_inactive[::skip], truth_copy_inactive[::skip, i], label=r'$'+labels[i]+r'$', color=colors[1])
            else:
                plt.plot(truth_t_copy[::skip], truth_copy[::skip, i], label=r'$'+labels[i]+r'$', color=colors[1])

        if "_" in labels[i]:
            plt.plot(est_t[::skip], estimate[::skip, i], label=r'$\hat{' + labels[i].split("_")[0] + r'}' + r'_' + labels[i].split("_")[1] + r'$', color=colors[0])
        else:
            plt.plot(est_t[::skip], estimate[::skip, i], label=r'$\hat{' + labels[i] + r'}$', color=colors[0])
        if isinstance(cov, np.ndarray):
            try:
                plt.plot(est_t[::skip], estimate[::skip, i].flatten() + 2 * cov_copy[::skip, i, i].flatten(), 'k-', alpha=0.5)
                plt.plot(est_t[::skip], estimate[::skip, i].flatten() - 2 * cov_copy[::skip, i, i].flatten(), 'k-', alpha=0.5)
            except:
                debug = 1

        plt.xlim(plot_start, plot_end)
        if ylim is not None: plt.ylim(ylim[0], ylim[1])
        plt.legend()
        if i == 0:
            plt.title(title)

    if show:
        plt.show()

    if save:
        title = title.translate(None, r'!@#${}/\\')
        global figure_directory
        if subdir is not None:
            filename = figure_directory + '/' + subdir + '/' + title + '.svg'
        else:
            filename = figure_directory + '/' + title + '.svg'
        plt.savefig(filename , bbox_inches='tight')
        plt.close()


def plot_cube(q_I_b, zetas, zeta_truth):

    cube = np.array([[-1., -1., -1.],
                     [1., -1., -1. ],
                     [1., 1., -1.],
                     [-1., 1., -1.],
                     [-1., -1., 1.],
                     [1., -1., 1. ],
                     [1., 1., 1.],
                     [-1., 1., 1.]])
    for i in range(8):
        cube[i,:,None] = q_I_b.R.dot(0.3*cube[i,:,None])

    right = [[cube[2],cube[3],cube[7],cube[6]]]
    front = [[cube[1],cube[2],cube[6],cube[5]]]
    bottom = [[cube[4],cube[5],cube[6],cube[7]]]
    rest = [[cube[0],cube[1],cube[2],cube[3]],
             [cube[0],cube[1],cube[5],cube[4]],
             [cube[4],cube[7],cube[3],cube[0]]]
    fig = plt.figure(10, figsize=(10,10))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(front,
     facecolors='red', linewidths=1, edgecolors='r', alpha=.25))
    ax.add_collection3d(Poly3DCollection(right,
     facecolors='green', linewidths=1, edgecolors='r', alpha=.25))
    ax.add_collection3d(Poly3DCollection(bottom,
     facecolors='blue', linewidths=1, edgecolors='r', alpha=.25))
    ax.add_collection3d(Poly3DCollection(rest,
     facecolors='grey', linewidths=1, edgecolors='r', alpha=.25))

    for i, (z, z_t) in enumerate(zip(zetas, zeta_truth)):
        z_NWU = q_I_b.R.T.dot(z)
        ax.plot([0, z_NWU[0]], [0, z_NWU[1]], [0, z_NWU[2]], '-b', label="est")
        zt_NWU = q_I_b.R.T.dot(z_t)
        ax.plot([0, zt_NWU[0]], [0, zt_NWU[1]], [0, zt_NWU[2]], '-r', label="truth")
        if i == 0:
            plt.legend()

    plt.ion()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    plt.show()
    plt.ioff()
    plt.pause(0.001)

def plot_3d_trajectory(ground_truth, feat_time=None, qzetas=None, depths=None, ids=None, p_b_c=np.zeros((3,1)), q_b_c=Quaternion.Identity(), fighandle=None):
    pos_time = ground_truth[:,0]
    position = ground_truth[:,1:4]
    orientation = ground_truth[:,4:8]
    if fighandle is not None:
        ax = fighandle
    else:
        plt.figure(2, figsize=(14,10))
        ax = plt.subplot(111, projection='3d')
    plt.plot(position[:1, 0],
             position[:1, 1],
             position[:1, 2], 'kx')
    plt.plot(position[:, 0],
             position[:, 1],
             position[:, 2], 'c-')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    e_x = 0.1*np.array([[1., 0, 0]]).T
    e_y = 0.1*np.array([[0, 1., 0]]).T
    e_z = 0.1*np.array([[0, 0, 1.]]).T
    if qzetas is not None:
        id_indices = np.unique(np.hstack(ids))
        colors = get_colors(len(id_indices), plt.cm.jet)
    for i in range(len(position)/25):
        j = i*25
        q_i_b = Quaternion(orientation[j,:,None])
        body_pos = position[j,:,None]
        x_end = body_pos + q_i_b.rot(e_x)
        y_end = body_pos + q_i_b.rot(e_y)
        z_end = body_pos + q_i_b.rot(e_z)
        plt.plot([body_pos[0, 0], x_end[0, 0]], [body_pos[1, 0], x_end[1, 0]], [body_pos[2, 0], x_end[2, 0]], 'r-')
        plt.plot([body_pos[0, 0], y_end[0, 0]], [body_pos[1, 0], y_end[1, 0]], [body_pos[2, 0], y_end[2, 0]], 'g-')
        plt.plot([body_pos[0, 0], z_end[0, 0]], [body_pos[1, 0], z_end[1, 0]], [body_pos[2, 0], z_end[2, 0]], 'b-')
        if feat_time is not None:
            feat_idx = np.argmin(feat_time < pos_time[j])
            camera_pos = body_pos + q_i_b.invrot(p_b_c)
            for qz, d, id in zip(qzetas[feat_idx], depths[feat_idx], ids[feat_idx]):
                if np.isfinite(qz).all() and np.isfinite(d):
                    q_c_z = Quaternion(qz[:,None])
                    zeta_end = camera_pos + q_i_b.rot(q_b_c.rot(q_c_z.rot(10.*e_z*d)))
                    plt.plot([camera_pos[0, 0], zeta_end[0, 0]], [camera_pos[1, 0], zeta_end[1, 0]], [camera_pos[2, 0], zeta_end[2, 0]], '--', color=colors[np.where(id_indices == id)[0][0]], lineWidth='1.0')

def play_trajectory_with_sim_features(image_time, image_data, feat_time, ids, lambdas):
    id_indices = np.unique(np.hstack(ids))
    colors = get_colors(len(id_indices), plt.cm.jet)
    out = cv2.VideoWriter('plots/sim_features.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640, 480))
    for t, img in zip(image_time, image_data):
        feat_idx = np.argmin(feat_time < t)
        frame_feats = lambdas[feat_idx]
        frame_ids = ids[feat_idx]
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for feat, id in zip(frame_feats, frame_ids):
            cv2.circle(img_color, tuple(feat.astype('uint16').tolist()), 5, colors[np.where(id_indices == id)[0][0]].tolist(), -1)
            cv2.putText(img_color, str(id), tuple(feat.astype('uint16').tolist()), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        cv2.imshow("simulated features", img_color)
        out.write(img_color)
        cv2.waitKey(33)
    out.release()
    cv2.destroyAllWindows()

