import vi_ekf as viekf
from tqdm import tqdm
from plot_helper import plot_side_by_side, init_plots, plot_3d_side_by_side, plot_3d_trajectory
from data import ROSbagData, History
from pyquat import Quaternion, quat_arr_to_euler
from math_helper import e_z
import numpy as np

start = 13.0
end = 30.0
# start = 15.0
# end = 16.0
# data = ROSbagData(filename='data/truth_imu_flight.bag', start=30.0, end=68.0, sim_features=True, load_new=True)
# data = ROSbagData(filename='data/xtion_collect.bag', start=start, end=end, sim_features=False, load_new=True, show_video=True)
data = ROSbagData(filename='~/truth_laser_xtion_imu.bag', start=start, end=end, sim_features=False, load_new=True, show_video=True)
# data = ROSbagData(filename='data/hand_carried/roll.bag', start=start, end=end, sim_features=True, load_new=True)
data.__test__()

ekf = viekf.VI_EKF(data.x0)
ekf.set_camera_to_IMU(data.data['p_b_c'], data.data['q_b_c'])
ekf.set_camera_intrinsics(data.data['cam_center'], data.data['cam_F'])
h = History()

for i, (t, pos, vel, att, gyro, acc, lambdas, depths, ids, qzetas) in enumerate(tqdm(data)):
    h.store(t, pos=pos, vel=vel, att=att)
    h.store(t, gyro=gyro, acc=acc)
    h.store(t, lambdas=lambdas, depths=depths, ids=ids)

    # propagate if this tick contains data
    if acc is not None and gyro is not None:
        x_hat, P = ekf.propagate(acc, gyro, t)
        h.store(t, x_hat=x_hat[:viekf.xZ], P=P[:viekf.dxZ, :viekf.dxZ])
        vel_c_i, omega_c_i = ekf.get_camera_state()
        h.store(t, vel_c_i=vel_c_i, omega_c_i=omega_c_i)

    # sensor updates - save off residual information
    if pos is not None:
        h.store(t, alt_res=ekf.update(-pos[2], 'alt', data.R['alt'], passive=False)[0])
        h.store(t, att_res=ekf.update(att, 'att', data.R['att'], passive=False)[0],
                pos_res=ekf.update(pos, 'pos', data.R['pos'], passive=True)[0],
                vel_res=ekf.update(vel, 'vel', data.R['vel'], passive=True)[0])

    if acc is not None:
        h.store(t, acc_res=ekf.update(acc[:2], 'acc', data.R['acc'], passive=False)[0])

    # feature updates
    if ids is not None and len(ids) > 0:
        ekf.keep_only_features(ids)
        for j, (l, depth, id, qz) in enumerate(zip(lambdas, depths, ids, qzetas)):
            lambda_res, lambda_hat = ekf.update(l, 'feat', data.R['lambda'], passive=False, i=id, depth=depth)
            depth_res, depth_hat = ekf.update(depth, 'depth', data.R['depth'], passive=np.isnan(depth), i=id)
            h.store(t, id, depth_hat=depth_hat, depth=depth, depth_res=depth_res)
            h.store(t, id, lambda_res=lambda_res, lambda_hat=lambda_hat, lmda=l)
            h.store(t, id, qzeta=qz, qzeta_hat=ekf.get_qzeta(id))
            h.store(t, id, zeta=Quaternion(qz).rot(e_z), zeta_hat=ekf.get_zeta(id))
            dxRHO_i = viekf.dxZ+3*ekf.global_to_local_feature_id[id]+2
            h.store(t, id, Pfeat=ekf.P[dxRHO_i, dxRHO_i,None,None])
        h.store(t, ids=ids)

# plot
if True:
    # convert all our linked lists to contiguous numpy arrays and initialize the plot parameters
    h.tonumpy()
    init_plots(start, end)
    print "plotting"

    plot_side_by_side('x_pos', viekf.xPOS, viekf.xPOS+3, h.t.x_hat, h.x_hat, cov=h.P, truth_t=h.t.pos, truth=h.pos, labels=['x', 'y', 'z'], start_t=start, end_t=end)
    plot_side_by_side('x_vel', viekf.xVEL, viekf.xVEL+3, h.t.x_hat, h.x_hat, cov=h.P, truth_t=h.t.vel, truth=h.vel, labels=['x', 'y', 'z'], start_t=start, end_t=end)
    # plot_side_by_side('x_att', viekf.xATT, viekf.xATT+4, h.t.x_hat, h.x_hat, truth_t=h.t.att, truth=h.att, labels=['w', 'x', 'y', 'z'])

    plot_side_by_side('x_euler', 0, 3, h.t.x_hat, quat_arr_to_euler(h.x_hat[:, viekf.xATT:viekf.xATT+4, 0].T).T, truth_t=h.t.att, truth=quat_arr_to_euler(h.att.T[0]).T, labels=[r'$\phi$', r'$\rho$', r'$\psi$'], start_t=start, end_t=end)
    # plot_side_by_side('x_b_g', viekf.xB_G, viekf.xB_G + 3, h.t.x_hat, h.x_hat, cov=h.P, labels=['x', 'y', 'z'], cov_bounds=(viekf.dxB_G,viekf.dxB_G+3))
    # plot_side_by_side('x_b_a', viekf.xB_A, viekf.xB_A + 3, h.t.x_hat, h.x_hat, cov=h.P, labels=['x', 'y', 'z'], cov_bounds=(viekf.dxB_A,viekf.dxB_A+3))
    # plot_side_by_side('x_mu', viekf.xMU, viekf.xMU+1, h.t.x_hat, h.x_hat, cov=h.P, labels=['mu'], cov_bounds=(viekf.dxMU,viekf.dxMU+1))

    # plot_side_by_side('z_alt_residual', 0, 1, h.t.alt_res, h.alt_res, labels=['z_alt_res'])
    # plot_side_by_side('z_att_residual', 0, 3, h.t.att_res, h.att_res, labels=['x', 'y', 'z'])
    # plot_side_by_side('z_acc_residual', 0, 2, h.t.acc_res, h.acc_res, labels=['x', 'y'])
    # plot_side_by_side('z_pos_residual', 0, 3, h.t.pos_res, h.pos_res, labels=['x', 'y', 'z'])
    # plot_side_by_side('z_vel_residual', 0, 3, h.t.vel_res, h.vel_res, labels=['x', 'y', 'z'])

    plot_side_by_side('u_gyro', 0, 3, h.t.gyro, h.gyro, labels=['x', 'y', 'z'], start_t=start, end_t=end)
    # plot_side_by_side('u_acc', 0, 3, h.t.acc, h.acc, labels=['x', 'y', 'z'])

    # plot_side_by_side('camera_vel', 0, 3, h.t.vel_c_i, h.vel_c_i, labels=['x','y','z'])
    # plot_side_by_side('camera_omega', 0, 3, h.t.omega_c_i, h.omega_c_i, labels=['x', 'y', 'z'])

    ids = []
    for step_ids in h.ids:
        for id in step_ids:
            if id not in ids:
                ids.append(id)

    for i in tqdm(ids):
        plot_side_by_side('lambda/x_{}'.format(i), 0, 2, h.t.lambda_hat[i], h.lambda_hat[i], truth_t=h.t.lmda[i], truth=h.lmda[i], labels=['u','v'], start_t=start, end_t=end)
        plot_side_by_side('zeta/x_{}'.format(i), 0, 3, h.t.zeta_hat[i], h.zeta_hat[i], truth_t=h.t.zeta[i], truth=h.zeta[i], labels=['x', 'y', 'z'], start_t=start, end_t=end)
        plot_side_by_side('qzeta/x_{}'.format(i), 0, 4, h.t.qzeta_hat[i], h.qzeta_hat[i], truth_t=h.t.qzeta[i], truth=h.qzeta[i], labels=['w', 'x', 'y', 'z'], start_t=start, end_t=end)
        # plot_side_by_side('x_feat_{}'.format(i), 0, 3, h.t.zeta_hat[i], h.zeta_hat[i], truth_t=h.t.zeta[i], truth=h.zeta[i], labels=['x', 'y', 'z'])
        # plot_side_by_side('x_qfeat_{}'.format(i), 0, 4, h.t.qzeta_hat[i], h.qzeta_hat[i], truth_t=h.t.qzeta[i], truth=h.qzeta[i], labels=['w','x', 'y', 'z'])
        # plot_side_by_side('lambda_{}_residual'.format(i), 0, 2, h.t.lambda_res[i], h.lambda_res[i], labels=['x', 'y'])

        if i in h.depth.keys():
            plot_side_by_side('rho/x_{}'.format(i), 0, 1, h.t.depth_hat[i], h.depth_hat[i], truth_t=h.t.depth[i], truth=h.depth[i], labels=[r'$\frac{1}{\rho}$'], start_t=start, end_t=end)
        else:
            plot_side_by_side('rho/x_{}'.format(i), 0, 1, h.t.depth_hat[i], h.depth_hat[i], labels=[r'$\frac{1}{\rho}$'], start_t=start, end_t=end)
            #
            #
            # plot_side_by_side('z_depth_{}_residual'.format(i), 0, 1, h.t.depth_res[i], h.depth_res[i], labels=['rho'])
quit()
    # Plot the 3d Trajectory and Estimate
    # plot_3d_side_by_side(h, ekf.p_b_c, ekf.q_b_c)
