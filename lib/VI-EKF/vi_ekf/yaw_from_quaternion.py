from pyquat import Quaternion
from math_helper import norm, skew
import numpy as np

# for i in range(100):
q = Quaternion.random()
# q = Quaternion.from_axis_angle(np.array([[0, 0, 1]]).T, np.pi/2.)
yaw_true = q.yaw
v = np.array([[0, 0, 1]]).T
beta = q.elements[1:]
beta /= norm(beta)
alpha = 2.*np.arccos(q.elements[0,0])
yaw_steven = 2.*np.arctan(beta.T.dot(v) * np.tan(alpha/2.))

w = q.rot(v)
s = v.T.dot(w)
delta = skew(v).dot(w)
qhat = Quaternion.exp(s*delta)
qstar = q * qhat.inverse
yaw_superjax = qstar.yaw

print "superjax", (yaw_superjax)
print "steven", (yaw_steven)
print "true", (yaw_true)
    # assert abs(yaw_true - yaw_test) < 1e-8, "wrong: true = %f, test = %f" % (yaw_true, yaw_test)