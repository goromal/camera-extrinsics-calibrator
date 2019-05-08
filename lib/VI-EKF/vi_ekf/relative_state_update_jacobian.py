from pyquat import Quaternion, norm, skew
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

e = 1e-6

## Scratch remove axis of rotation from quaternion
u = np.array([[0, 0, 1.]]).T
# u = np.random.random((3,1))
u /= norm(u)

psi_hist, yaw_hist, s_hist, v_hist, qz_hist, qx_hist, qy_hist = [], [], [], [], [], [], []
for i in tqdm(range(10000)):
    # qm0 = Quaternion.from_euler(0.0, 10.0, 2*np.pi*np.random.random() - np.pi)
    # qm0 = Quaternion.from_euler(0.0, np.pi*np.random.random() - np.pi/2.0, 0.0)
    qm0 = Quaternion.from_euler(2*np.pi*np.random.random() - np.pi, np.pi * np.random.random() - np.pi / 2.0, 0.0)
    # qm0 = Quaternion.from_euler(270.0 * np.pi / 180.0, 85.0 * np.pi / 180.0, 90.0 * np.pi / 180.0)
    # qm0 = Quaternion.random()

    w = qm0.rot(u)
    th = u.T.dot(w)
    ve = skew(u).dot(w)
    qp0 = Quaternion.exp(th * ve)

    epsilon = np.eye(3) * e

    t = u.T.dot(qm0.rot(u))
    v = skew(u).dot(qm0.rot(u))

    tv0 = u.T.dot(qm0.rot(u)) * (skew(u).dot(qm0.rot(u)))
    a_dtvdq = -v.dot(u.T.dot(qm0.R.T).dot(skew(u))) - t*skew(u).dot(qm0.R.T.dot(skew(u)))
    d_dtvdq = np.zeros_like(a_dtvdq)

    nd = norm(t * v)
    d0 = t * v
    qd0 = Quaternion.exp(d0)
    sk_tv = skew(t * v)
    Tau = (np.eye(3) + ((1. - np.cos(t)) * sk_tv) / (t * t) + ((t - np.sin(t)) * sk_tv.dot(sk_tv)) / (t * t * t)).T
    Tau_approx = np.eye(3) + 1. / 2. * sk_tv
    a_dqdq = Tau.dot(a_dtvdq)
    a_dqdq_approx = Tau_approx.dot(a_dtvdq)
    d_dqdq = np.zeros_like(a_dqdq)

    a_dexpdd = Tau
    d_dexpdd = np.zeros_like(a_dexpdd)

    psi_hist.append(qm0.yaw)
    yaw_hist.append(qd0.yaw)
    s_hist.append(t)
    v_hist.append(v)
    qz_hist.append(qd0.z)
    qy_hist.append(qd0.y)
    qx_hist.append(qd0.x)

    for i in range(3):
        qmi = qm0 + epsilon[:, i, None]
        w = qmi.rot(u)
        th = u.T.dot(w)
        ve = skew(u).dot(w)
        qpi = Quaternion.exp(th * ve)
        d_dqdq[:, i, None] = (qpi - qp0) / e

        qdi = Quaternion.exp(d0 + epsilon[:, i, None])
        d_dexpdd[:, i, None] = (qdi - qd0) / e

        tvi = u.T.dot(qmi.rot(u)) * (skew(u).dot(qmi.rot(u)))
        d_dtvdq[:, i, None] = (tvi - tv0) / e

    total_diff = np.sum(np.abs(a_dqdq - d_dqdq))

    if total_diff > 5e-2:
        print "analytical:\n", np.around(a_dqdq, 5)
        print "approx:\n", np.around(a_dqdq_approx, 5)
        print "finite difference:\n", np.around(d_dqdq, 5)

        # print "dan:\n", np.around(N,5)

        print "bonus A:\n", np.around(a_dtvdq, 5)
        print "bonus FD:\n", np.around(d_dtvdq, 5)
        print "bonus diff:\n", np.sum(a_dtvdq - d_dtvdq)
        # #
        print "magic A:\n", np.around(Tau, 5)
        print "magic FD:\n", np.around(d_dexpdd, 5)
        print "magic diff:\n", np.sum(Tau - d_dexpdd)
        print "total diff:\n", total_diff, 5
# plt.scatter(psi_hist, yaw_hist)
# plt.xlabel("psi")
# plt.ylabel("yaw+")
# plt.show()
# plt.scatter(s_hist, yaw_hist)
# plt.xlabel("s")
# plt.ylabel("yaw+")
# plt.show()
# plt.scatter(qx_hist, yaw_hist)
# plt.xlabel("qx")
# plt.ylabel("yaw+")
# plt.show()
# plt.scatter(qy_hist, yaw_hist)
# plt.xlabel("qy")
# plt.ylabel("yaw+")
# plt.show()
# plt.scatter(qz_hist, yaw_hist)
# plt.xlabel("qz")
# plt.ylabel("yaw+")
# plt.show()

