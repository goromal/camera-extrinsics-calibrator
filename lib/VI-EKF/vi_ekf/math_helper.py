from pyquat import Quaternion
import numpy as np

def norm(v, axis=None):
    return np.sqrt(np.sum(v*v, axis=axis))

cross_matrix = np.array([[[0, 0, 0],
                          [0, 0, -1.0],
                          [0, 1.0, 0]],
                         [[0, 0, 1.0],
                          [0, 0, 0],
                          [-1.0, 0, 0]],
                         [[0, -1.0, 0],
                          [1.0, 0, 0],
                          [0, 0, 0]]])

e_z = np.array([[0, 0, 1.0]]).T

# Creates the skew-symmetric matrix from v
def skew(v):
    # assert v.shape[0] == 3
    return cross_matrix.dot(v).squeeze()

# Creates 3x2 projection matrix onto the plane perpendicular to zeta
e_x_e_y = np.array([[1., 0, 0], [0, 1., 0]]).T
def T_zeta(q_zeta):
    # assert isinstance(q_zeta, Quaternion)
    # The coordinate basis normal to the feature vector, expressed in the camera frame (hence the active rotation)
    # (This is where the Lie Group is linearized about)
    return q_zeta.rot(e_x_e_y)

# Finds the difference between two feature quaternions
def q_feat_boxminus(qi, qj):
    # assert isinstance(qi, Quaternion) and isinstance(qj, Quaternion)

    zetai = qi.rot(e_z)
    zetaj = qj.rot(e_z)

    if norm(zetai - zetaj) > 1e-16:
        zj_x_zi = skew(zetaj).dot(zetai)
        s = zj_x_zi / norm(zj_x_zi) # The vector about which rotation occurs (normalized)
        theta = np.arccos(zetaj.T.dot(zetai)) # the magnitude of the rotation
        # The rotation vector exists in the plane normal to the feature vector.  Therefore if we rotate to this
        # basis, then all the information is stored in the x and y components only.  This reduces the dimensionality
        # of the delta-feature quaternion
        dq = theta * T_zeta(qj).T.dot(s)
        return dq
    else:
        return np.zeros((2,1))


# Adds a delta feature quaternion to quaternion q, returns a quaternion
def q_feat_boxplus(q, dq):
    # assert isinstance(q, Quaternion) and dq.shape == (2,1)
    return Quaternion.exp(T_zeta(q).dot(dq)) * q
    # zeta = q_fast.rot(e_z)
    # return Quaternion.from_two_unit_vectors(e_z, zeta)


# Calculates the quaternion which rotates u into v.
# That is, if q = q_from_two_unit_vectors(u,v)
# q.rot(u) = v and q.invrot(v) = u
# This is a vectorized version which returns multiple quaternions for multiple v's from one u
def q_array_from_two_unit_vectors(u, v):
    # assert u.shape[0] == 3
    # assert v.shape[0] == 3
    u = u.copy()
    v = v.copy()

    num_arrays = v.shape[1]
    arr = np.vstack((np.ones((1, num_arrays)), np.zeros((3, num_arrays))))

    d = u.T.dot(v)

    invs = (2.0*(1.0+d))**-0.5
    xyz = skew(u).dot(v)*invs
    arr[0, :, None] = 0.5 / invs.T
    arr[1:,:] = xyz
    arr /= norm(arr, axis=0)
    return arr

def run_tests():
    # run some math helper tests

    # Test vectorized quat from two unit vectors
    v1 = np.random.uniform(-1, 1, (3, 100))
    v2 = np.random.uniform(-1, 1, (3, 1))
    v3 = np.random.uniform(-1, 1, (3, 1))
    v1 /= norm(v1, axis=0)
    v2 /= norm(v2)
    v3 /= norm(v3)
    # On a single vector
    assert norm(Quaternion(q_array_from_two_unit_vectors(v3, v2)).rot(v3) - v2) < 1e-8
    # on a bunch of vectors
    quat_array = q_array_from_two_unit_vectors(v2, v1)
    for q, v in zip(quat_array.T, v1.T):
        Quaternion(q[:,None]).rot(v2) - v[:,None]

    # Test T_zeta
    q2 = q_array_from_two_unit_vectors(e_z, v2)
    assert norm(T_zeta(Quaternion(q2)).T.dot(v2)) < 1e-8

    # Check derivative of T_zeta - This was giving me trouble
    d_dTdq = np.zeros((2,2))
    q = Quaternion(np.random.uniform(-1, 1, (4,1)))
    q.arr[3] = 0.0
    q.normalize()
    x0 = T_zeta(q).T.dot(v2)
    epsilon = 1e-6
    I = np.eye(2)*epsilon
    for i in range(2):
        qplus = q_feat_boxplus(q, I[:,i,None])
        xprime = T_zeta(qplus).T.dot(v2)
        d_dTdq[i, :, None] = (xprime - x0)/epsilon
    a_dTdq = -T_zeta(q).T.dot(skew(v2).dot(T_zeta(q)))
    assert (abs(a_dTdq - d_dTdq) < 1e-6).all()

    # Check Derivative  dqzeta/dqzeta <- this was also giving me trouble
    for j in range(1000):
        d_dqdq = np.zeros((2,2))
        if j ==0:
            q = Quaternion.Identity()
        else:
            q = Quaternion(np.random.uniform(-1, 1, (4, 1)))
            q.arr[3] = 0.0
            q.normalize()
        for i in range(2):
            d_dqdq[i,:,None] = q_feat_boxminus(q_feat_boxplus(q, I[:,i,None]), q)/epsilon
        a_dqdq = T_zeta(q).T.dot(T_zeta(q))
        assert (abs(a_dqdq - d_dqdq) < 1e-1).all()



    # Check Manifold Consistency
    for i in range(1000):
        omega = np.random.uniform(-1, 1, (3, 1))
        omega2 = np.random.uniform(-1, 1, (3, 1))
        omega[2] = 0.0
        omega2[2] = 0.0
        x = Quaternion.exp(omega)
        y = Quaternion.exp(omega2)
        dx = np.random.normal(0.0, 0.5, (2,1))

        # Check x [+] 0 == x
        assert norm( q_feat_boxplus(x, np.zeros((2,1))) - x) < 1e-8

        # Check x [+] (y [-] x) == y (compare the rotated zetas, because there are infinitely
        # many quaternions which return the same zeta.)  We don't have the framework to handle
        # forcing the quaternion to actually be the same
        assert norm( (q_feat_boxplus(x, q_feat_boxminus(y, x))).rot(e_z) - y.rot(e_z)) < 1e-8

        # Check (x [+] dx) [-] x == dx
        assert norm( q_feat_boxminus(q_feat_boxplus(x, dx), x) - dx) < 1e-8
    # assert norm( q_feat_boxminus(q_feat_boxplus(qzeta, dqzeta), qzeta) - dqzeta) < 1e-8

    print "math helper test: [PASSED]"

if __name__ == '__main__':
    run_tests()




