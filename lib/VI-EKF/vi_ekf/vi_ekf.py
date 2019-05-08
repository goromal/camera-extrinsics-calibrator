import numpy as np
from pyquat import Quaternion
from math_helper import skew, T_zeta, norm, q_feat_boxminus, q_feat_boxplus
import scipy.linalg

dxPOS = 0
dxVEL = 3
dxATT = 6
dxB_A = 9
dxB_G = 12
dxMU = 15
dxZ = 16

xPOS = 0
xVEL = 3
xATT = 6
xB_A = 10
xB_G = 13
xMU = 16
xZ = 17

uA = 0
uG = 3

I_2x3 = np.array([[1, 0, 0],
                  [0, 1, 0]])
I_3x3 = np.eye(3)
I_2x2 = np.eye(2)

class VI_EKF():
    def __init__(self, x0, multirotor=True):
        # assert x0.shape == (xZ, 1)

        # 17 main states + 5N feature states
        # pos, vel, att, b_gyro, b_acc, mu, q_feat, rho_feat, q_feat, rho_feat ...
        self.x = x0
        self.u = np.zeros((6,1))

        # Process noise matrix for the 16 main delta states
        self.Qx = np.diag([0.001, 0.001, 0.001,     # pos
                           0.1, 0.1, 0.1,           # vel
                           0.005, 0.005, 0.005,     # att
                           1e-7, 1e-7, 1e-7,        # b_acc
                           1e-8, 1e-8, 1e-8,        # b_omega
                           0.0])                    # mu

        # process noise matrix for the features (assumed all the same) 3x3
        self.Qx_feat = np.diag([0.001, 0.001, 0.01]) # x, y, and 1/depth

        # Process noise assumed from inputs (mechanized sensors)
        self.Qu = np.diag([0.05, 0.05, 0.05,        # y_acc
                           0.001, 0.001, 0.001])    # y_omega



        # State covariances.  Size is (16 + 3N) x (16 + 3N) where N is the number of
        # features currently being tracked
        self.P = np.diag([0.0001, 0.0001, 0.0001,   # pos
                           0.01, 0.01, 0.01,        # vel
                           0.001, 0.001, 0.001,     # att
                           1e-2, 1e-2, 1e-3,        # b_acc
                           1e-3, 1e-3, 1e-3,        # b_omega
                           1e-7])                   # mu

        # Initial Covariance estimate for new features
        self.P0_feat = np.diag([0.01, 0.01, 0.1]) # x, y, and 1/depth

        # gravity vector (NED)
        self.gravity = np.array([[0, 0, 9.80665]]).T

        # Unit vectors in the x, y, and z directions (used a lot for projection functions)
        self.ihat = np.array([[1, 0, 0]]).T
        self.jhat = np.array([[0, 1, 0]]).T
        self.khat = np.array([[0, 0, 1]]).T

        # The number of features currently being tracked
        self.len_features = 0

        # The next feature id to be assigned to a feature
        self.next_feature_id = 0

        # Set of initialized feature ids
        self.initialized_features = set()
        self.global_to_local_feature_id = {}

        # Body-to-Camera transform
        self.q_b_c = Quaternion(np.array([[1, 0, 0, 0]]).T) # Rotation from body to camera
        self.p_b_c = np.array([[0, 0, 0]]).T # translation from body to camera (in body frame)

        self.measurement_functions = dict()
        self.measurement_functions['acc'] = self.h_acc
        self.measurement_functions['alt'] = self.h_alt
        self.measurement_functions['att'] = self.h_att
        self.measurement_functions['pos'] = self.h_pos
        self.measurement_functions['vel'] = self.h_vel
        self.measurement_functions['qzeta'] = self.h_qzeta
        self.measurement_functions['feat'] = self.h_feat
        self.measurement_functions['pixel_vel'] = self.h_pixel_vel
        self.measurement_functions['depth'] = self.h_depth
        self.measurement_functions['inv_depth'] = self.h_inv_depth

        # Matrix Workspace
        self.A = np.zeros((dxZ, dxZ))
        self.G = np.zeros((dxZ, 6))
        self.I_big = np.eye(dxZ)
        self.dx = np.zeros((dxZ, 1))

        self.use_drag_term = True
        self.default_depth = np.array([[1.5]])

        self.last_propagate = None
        self.cam_center = np.array([[320.0, 240.0]]).T
        self.cam_F = np.array([[250.0, 250.0]]).dot(I_2x3)
        self.use_drag_term = multirotor

    def set_camera_intrinsics(self, center, F):
        assert center.shape == (2,1) and F.shape ==(2,3)
        self.cam_center = center
        self.cam_F = F

    def set_camera_to_IMU(self, translation, rotation):
        # assert translation.shape == (3,1) and isinstance(rotation, Quaternion)
        self.p_b_c = translation
        self.q_b_c = rotation

    # Returns the depth to all features
    def get_depths(self):
        return 1./self.x[xZ+4::5]

    # Returns the estimated bearing vector to all features
    def get_zetas(self):
        zetas = np.zeros((3, self.len_features))
        for i in range(self.len_features):
            qzeta = self.x[xZ + 5 * i:xZ + 5 * i + 4, :]  # 4-vector quaternion
            zetas[:, i, None] = Quaternion(qzeta).rot(self.khat)  # 3-vector pointed at the feature in the camera frame
        return zetas

    # Returns the estimated bearing vector to a single feature with id i
    def get_zeta(self, i):
        ft_id = self.global_to_local_feature_id[i]
        qzeta = self.x[xZ + 5 * ft_id:xZ + 5 * ft_id + 4, :]  # 4-vector quaternion
        return Quaternion(qzeta).rot(self.khat)  # 3-vector pointed at the feature in the camera frame

    # Returns all quaternions which rotate the camera z axis to the bearing vectors directed at the tracked features
    def get_qzetas(self):
        qzetas = np.zeros((self.len_features, 4))
        for i in range(self.len_features):
            qzetas[i,:,None] = self.x[xZ+5*i:xZ+5*i+4]   # 4-vector quaternion
        return qzetas

    # Returns all quaternions which rotate the camera z axis to the bearing vectors directed at the tracked features
    def get_qzeta(self, i):
        ft_id = self.global_to_local_feature_id[i]
        return self.x[xZ + 5 * ft_id:xZ + 5 * ft_id + 4, :]  # 4-vector quaternion

    def get_camera_state(self):
        vel = self.x[xVEL:xVEL + 3]
        omega = self.u[uG:uG+3] - self.x[xB_G:xB_G+3]
        vel_c_i = self.q_b_c.invrot(vel + skew(omega).dot(self.p_b_c))
        omega_c_i = self.q_b_c.invrot(omega)
        return vel_c_i, omega_c_i

    # Adds the state with the delta state on the manifold
    def boxplus(self, x, dx):
        # assert  x.shape == (xZ+5*self.len_features, 1) and dx.shape == (dxZ+3*self.len_features, 1)

        out = np.zeros((xZ+5*self.len_features, 1))

        # Add position and velocity vector states
        out[xPOS:xPOS+3] = x[xPOS:xPOS+3] + dx[xPOS:xPOS+3]
        out[xVEL:xVEL+3] = x[xVEL:xVEL+3] + dx[xVEL:xVEL+3]

        # Add attitude quaternion state on the manifold
        out[xATT:xATT+4] = (Quaternion(x[xATT:xATT+4]) + dx[dxATT:dxATT+3]).elements

        # add bias and drag term vector states
        out[xB_A:xB_A+3] = x[xB_A:xB_A+3] + dx[dxB_A:dxB_A+3]
        out[xB_G:xB_G+3] = x[xB_G:xB_G+3] + dx[dxB_G:dxB_G+3]
        out[xMU] = x[xMU] + dx[dxMU]

        # add Feature quaternion states
        for i in range(self.len_features):
            xFEAT = xZ+i*5
            xRHO = xZ+i*5+4
            dxFEAT = dxZ+3*i
            dxRHO = dxZ+3*i+2
            dqzeta = dx[dxFEAT:dxRHO,:]  # 2-vector which is the derivative of qzeta
            qzeta = Quaternion(x[xFEAT:xFEAT+4,:]) # 4-vector quaternion

            # Feature Quaternion States (use manifold)
            out[xFEAT:xRHO,:] = q_feat_boxplus(qzeta, dqzeta).elements

            # Inverse Depth State
            out[xRHO,:] = x[xRHO] + dx[dxRHO]
        return out

    # propagates all states, features and covariances
    def propagate(self, y_acc, y_gyro, t):
        # assert y_acc.shape == (3, 1) and y_gyro.shape == (3, 1) and isinstance(t, float)

        if self.last_propagate is not None:
            # calculate dt from t
            dt = t - self.last_propagate
            self.u[uA:uA+3] = y_acc
            self.u[uG:uG+3] = y_gyro

            # Propagate
            xdot, A, G = self.dynamics(self.x, self.u)
            Pdot = A.dot(self.P) + self.P.dot(A.T) + G.dot(self.Qu).dot(G.T) + self.Qx
            self.x = self.boxplus(self.x, xdot * dt)
            self.P += Pdot*dt

        # Update last_propagate
        self.last_propagate = t

        return self.x.copy(), self.P.copy()

    def update(self, z, measurement_type, R, passive=False, **kwargs):
        # assert measurement_type in self.measurement_functions.keys(), "Unknown Measurement Type"

        passive_update = passive

        # If we haven't seen this feature before, then initialize it
        if measurement_type == 'feat':
            if kwargs['i'] not in self.global_to_local_feature_id.keys():
                self.init_feature(z, id=kwargs['i'], depth=(kwargs['depth'] if 'depth' in kwargs else np.nan))

        zhat, H = self.measurement_functions[measurement_type](self.x, **kwargs)

        # Calculate residual in the proper manner
        if measurement_type == 'qzeta':
            residual = q_feat_boxminus(Quaternion(z), Quaternion(zhat))
        elif measurement_type == 'att':
            residual = Quaternion(z) - Quaternion(zhat)
            if (abs(residual) > 1).any():
                residual = Quaternion(z) - Quaternion(zhat)
        else:
            residual = z - zhat



        # Perform state and covariance update
        if not passive_update:
            try:
                K = self.P.dot(H.T).dot(scipy.linalg.inv(R + H.dot(self.P).dot(H.T)))
                self.P = (self.I_big - K.dot(H)).dot(self.P)
                self.x = self.boxplus(self.x, K.dot(residual))
            except:
                print "Nan detected in", measurement_type, "update"

        return residual, zhat

    # Used for overriding imu biases, Not to be used in real life
    def set_imu_bias(self, b_g, b_a):
        # assert b_g.shape == (3,1) and b_a.shape == (3,1)
        self.x[xB_G:xB_G+3] = b_g
        self.x[xB_A:xB_A+3] = b_a

    # Used to initialize a new feature.  Returns the feature id associated with this feature
    def init_feature(self, l, id, depth=np.nan):
        # assert l.shape == (2, 1) and depth.shape == (1, 1)

        self.len_features += 1
        # self.feature_ids.append(self.next_feature_id)
        self.global_to_local_feature_id[id] = self.next_feature_id
        self.next_feature_id += 1

        # Adjust lambdas to be with respect to the center of the image
        l_centered = l - self.cam_center

        # Calculate Quaternion to feature
        f = self.cam_F[0,0]
        zeta = np.array([[ l_centered[0,0], l_centered[1,0], f]]).T
        zeta /= norm(zeta)
        q_zeta = Quaternion.from_two_unit_vectors(self.khat, zeta).elements

        if np.isnan(depth):
            # Best guess of depth without other information
            if self.len_features > 0:
                depth = np.average(1.0/self.x[xZ + 4::5])
            else:
                depth = self.default_depth
        self.x = np.vstack((self.x, q_zeta, 1./depth)) # add 5 states to the state vector

        # Add three states to the process noise matrix
        self.Qx = scipy.linalg.block_diag(self.Qx, self.Qx_feat)
        self.P = scipy.linalg.block_diag(self.P, self.P0_feat)

        # Adjust the matrix workspace allocation to fit this new feature
        self.A = np.zeros((dxZ + 3 * self.len_features, dxZ + 3 * self.len_features))
        self.G = np.zeros((dxZ + 3 * self.len_features, 6))
        self.I_big = np.eye(dxZ+3*self.len_features)
        self.dx = np.zeros((dxZ + 3 * self.len_features, 1))

        return self.next_feature_id - 1

    # Used to remove a feature from the EKF.  Removes the feature from the features array and
    # Clears the associated rows and columns from the covariance.  The covariance matrix will
    # now be 3x3 smaller than before and the feature array will be 5 smaller
    def clear_feature(self, id):
        local_feature_id = self.global_to_local_feature_id[id]
        xZETA_i = xZ + 5 * local_feature_id
        dxZETA_i = dxZ + 3 * local_feature_id
        del self.feature_ids[local_feature_id]
        self.len_features -= 1
        self.global_to_local_feature_id = {self.feature_ids[i] : i for i in range(len(self.feature_ids)) }

        # Create masks to remove portions of state and covariance
        xmask = np.ones_like(self.x, dtype=bool).squeeze()
        dxmask = np.ones_like(self.dx, dtype=bool).squeeze()
        xmask[xZETA_i:xZETA_i + 5] = False
        dxmask[dxZETA_i:dxZETA_i + 3] = False

        # Matrix Workspace Modifications
        self.x = self.x[xmask,...]
        self.P = self.P[dxmask, :][:, dxmask]
        self.dx = self.dx[dxmask,...]
        self.A = np.zeros_like(self.P)
        self.G = np.zeros((len(self.dx), 4))
        self.I_big = np.eye(len(self.dx))


    def keep_only_features(self, features):
        features_to_clear = self.initialized_features.difference(set(features))
        for f in features_to_clear:
            self.clear_feature(f)

    # Determines the derivative of state x given inputs u and Jacobian of state with respect to x and u
    # the returned value of f is a delta state, delta features, and therefore is a different
    # size than the state and features and needs to be applied with boxplus
    def dynamics(self, x, u):
        # assert x.shape == (xZ+5*self.len_features, 1) and u.shape == (6,1)

        # Reset Matrix Workspace
        self.dx.fill(0.0)
        self.A.fill(0.0)
        self.G.fill(0.0)

        vel = x[xVEL:xVEL+3]
        q_I_b = Quaternion(x[xATT:xATT+4])

        omega = u[uG:uG+3] - x[xB_G:xB_G+3]
        acc = u[uA:uA+3] - x[xB_A:xB_A+3]
        acc_z = np.array([[0, 0, acc[2,0]]]).T
        mu = x[xMU, 0]

        gravity_B = q_I_b.invrot(self.gravity)
        vel_I = q_I_b.invrot(vel)
        vel_xy = I_2x3.T.dot(I_2x3).dot(vel)


        # CALCULATE STATE DYNAMICS
        self.dx[dxPOS:dxPOS+3] = vel_I
        if self.use_drag_term:
            self.dx[dxVEL:dxVEL+3] =  acc_z + gravity_B - mu * vel_xy
        else:
            self.dx[dxVEL:dxVEL+3] = acc + gravity_B
        self.dx[dxATT:dxATT+3] = omega

        ###################################
        # STATE JACOBIAN
        self.A[dxPOS:dxPOS+3, dxVEL:dxVEL+3] = q_I_b.R
        self.A[dxPOS:dxPOS+3, dxATT:dxATT+3] = skew(vel_I)
        if self.use_drag_term:
            self.A[dxVEL:dxVEL+3, dxVEL:dxVEL+3] = - mu*I_2x3.T.dot(I_2x3)
            self.A[dxVEL:dxVEL + 3, dxB_A:dxB_A + 3] = -self.khat.dot(self.khat.T)
            self.A[dxVEL:dxVEL + 3, dxMU, None] = -vel_xy
        else:
            self.A[dxVEL:dxVEL + 3, dxB_A:dxB_A + 3] = -I_3x3
        self.A[dxVEL:dxVEL+3, dxATT:dxATT+3] = skew(gravity_B)
        self.A[dxATT:dxATT+3, dxB_G:dxB_G+3] = -I_3x3

        #################################
        ## INPUT JACOBIAN
        if self.use_drag_term:
            self.G[dxVEL:dxVEL+3, uA:uA+3] = self.khat.dot(self.khat.T)
        else:
            self.G[dxVEL:dxVEL + 3, uA:uA + 3] = I_3x3
        self.G[dxATT:dxATT+3, uG:uG+3] = I_3x3

        # Camera Dynamics
        omega_c_i = self.q_b_c.invrot(omega)
        vel_c_i = self.q_b_c.invrot(vel - skew(omega).dot(self.p_b_c))


        for i in range(self.len_features):
            xZETA_i = xZ+i*5
            xRHO_i = xZ+5*i+4
            dxZETA_i = dxZ + i*3
            dxRHO_i = dxZ + i*3+2

            q_zeta = Quaternion(x[xZETA_i:xZETA_i+4,:])
            rho = x[xRHO_i,0]
            zeta = q_zeta.rot(self.khat)
            T_z = T_zeta(q_zeta)
            skew_zeta = skew(zeta)
            skew_vel_c = skew(vel_c_i)
            skew_p_b_c = skew(self.p_b_c)
            R_b_c = self.q_b_c.R
            rho2 = rho*rho

            #################################
            ## FEATURE DYNAMICS
            self.dx[dxZETA_i:dxZETA_i+2,:] = -T_z.T.dot(omega_c_i + rho*skew_zeta.dot(vel_c_i))
            self.dx[dxRHO_i,:] = rho2*zeta.T.dot(vel_c_i)

            #################################
            ## FEATURE STATE JACOBIAN
            self.A[dxZETA_i:dxZETA_i+2, dxVEL:dxVEL+3] = -rho*T_z.T.dot(skew_zeta).dot(R_b_c)
            self.A[dxZETA_i:dxZETA_i+2, dxB_G:dxB_G+3] = T_z.T.dot(rho*skew_zeta.dot(R_b_c).dot(skew_p_b_c) + R_b_c)
            self.A[dxZETA_i:dxZETA_i+2, dxZETA_i:dxZETA_i+2] = -T_z.T.dot(skew(rho * skew_zeta.dot(vel_c_i) + omega_c_i) + (rho * skew_vel_c.dot(skew_zeta))).dot(T_z)
            self.A[dxZETA_i:dxZETA_i+2, dxRHO_i,None] = -T_z.T.dot(skew_zeta).dot(vel_c_i)
            self.A[dxRHO_i, dxVEL:dxVEL+3] = rho2*zeta.T.dot(R_b_c)
            self.A[dxRHO_i, dxB_G:dxB_G+3] = -rho2*zeta.T.dot(R_b_c).dot(skew_p_b_c)
            self.A[dxRHO_i, dxZETA_i:dxZETA_i+2] = -rho2*vel_c_i.T.dot(skew_zeta).dot(T_z)
            self.A[dxRHO_i, dxRHO_i] = 2*rho*zeta.T.dot(vel_c_i).squeeze()

            #################################
            ## FEATURE INPUT JACOBIAN
            self.G[dxZETA_i:dxZETA_i+2, uG:uG+3] = -T_z.T.dot(R_b_c + rho*skew_zeta.dot(R_b_c).dot(skew_p_b_c))
            self.G[dxRHO_i, uG:] = rho2*zeta.T.dot(R_b_c).dot(skew_p_b_c)

        return self.dx, self.A, self.G

    # Accelerometer model
    # Returns estimated measurement (2 x 1) and Jacobian (2 x 16+3N)
    def h_acc(self, x):
        # assert x.shape==(xZ+5*self.len_features,1)

        vel = x[xVEL:xVEL + 3]
        b_a = x[xB_A:xB_A + 3]
        mu = x[xMU, 0]

        h = I_2x3.dot(-mu*vel + b_a)

        dhdx = np.zeros((2, dxZ+3*self.len_features))
        dhdx[:,dxVEL:dxVEL+3] = -mu * I_2x3
        dhdx[:,dxB_A:dxB_A+3] = I_2x3
        dhdx[:,dxMU,None] = -I_2x3.dot(vel)

        return h, dhdx

    # Altimeter model
    # Returns estimated measurement (1x1) and Jacobian (1 x 16+3N)
    def h_alt(self, x):
        # assert x.shape == (xZ + 5 * self.len_features, 1)

        h = -x[xPOS+2,:,None]

        dhdx = np.zeros((1, dxZ+3*self.len_features))
        dhdx[0,dxPOS+2] = -1.0

        return h, dhdx

    # Attitude Model
    # Returns the estimated attitude measurement (4x1) and Jacobian (3 x 16+3N)
    def h_att(self, x, **kwargs):
        # assert x.shape == (xZ + 5 * self.len_features, 1)

        h = x[xATT:xATT + 4]

        dhdx = np.zeros((3, dxZ + 3 *self.len_features))
        dhdx[:,dxATT:dxATT+3] = I_3x3

        return h, dhdx

    # Position Model
    # Returns the estimated Position measurement (3x1) and Jacobian (3 x 16+3N)
    def h_pos(self, x):
        # assert x.shape == (xZ + 5 * self.len_features, 1)

        h = x[xPOS:xPOS + 3]

        dhdx = np.zeros((3, dxZ + 3 * self.len_features))
        dhdx[:, dxPOS:dxPOS + 3] = I_3x3

        return h, dhdx

    # Velocity Model
    # Returns the estimated Position measurement (3x1) and Jacobian (3 x 16+3N)
    def h_vel(self, x):
        # assert x.shape == (xZ + 5 * self.len_features, 1)

        h = x[xVEL:xVEL + 3]

        dhdx = np.zeros((3, dxZ + 3 * self.len_features))
        dhdx[:, dxVEL:dxVEL + 3] = I_3x3

        return h, dhdx

    # qzeta model for feature index i
    # Returns estimated qzeta (4x1) and Jacobian (3 x 16+3N)
    def h_qzeta(self, x, **kwargs):
        # assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int)
        i = self.global_to_local_feature_id[kwargs['i']]
        dxZETA_i = dxZ + i * 3
        q_c_z = x[xZ+i*5:xZ+i*5+4]

        h = q_c_z

        dhdx = np.zeros((2, dxZ+3*self.len_features))
        dhdx[:, dxZETA_i:dxZETA_i+2] = I_2x2

        return h, dhdx

    # Feature model for feature index i
    # Returns estimated pixel measurement (2x1) and Jacobian (2 x 16+3N)
    def h_feat(self, x, **kwargs):
        # assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int)
        i = self.global_to_local_feature_id[kwargs['i']]
        dxZETA_i = dxZ + i * 3
        q_zeta = Quaternion(x[xZ+i*5:xZ+i*5+4])

        zeta = q_zeta.rot(self.khat)
        sk_zeta = skew(zeta)
        ezT_zeta = (self.khat.T.dot(zeta)).squeeze()
        T_z = T_zeta(q_zeta)

        h = self.cam_F.dot(zeta)/ezT_zeta + self.cam_center

        dhdx = np.zeros((2, dxZ+3*self.len_features))
        dhdx[:, dxZETA_i:dxZETA_i+2] = -self.cam_F.dot((sk_zeta.dot(T_z))/ezT_zeta - zeta.dot(self.khat.T).dot(sk_zeta).dot(T_z)/(ezT_zeta*ezT_zeta))

        return h, dhdx

    # Feature depth measurement
    # Returns estimated measurement (1x1) and Jacobian (1 x 16+3N)
    def h_depth(self, x, i):
        # assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int) and i in self.feature_ids
        local_id = self.global_to_local_feature_id[i]
        rho = x[xZ+local_id*5+4,0]

        h = np.array([[1.0/rho]])

        dhdx = np.zeros((1, dxZ+3*self.len_features))
        dhdx[0, dxZ+3*local_id+2,None] = -1/(rho*rho)

        return h, dhdx

    # Feature inverse depth measurement
    # Returns estimated measurement (1x1) and Jacobian (1 x 16+3N)
    def h_inv_depth(self, x, i):
        # assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int)
        h = x[xZ+i*5+4,None]

        dhdx = np.zeros((1, dxZ+3*self.len_features))
        dhdx[0, dxZ+3*i+2] = 1

        return h, dhdx

    # Feature pixel velocity measurement
    # Returns estimated measurement (2x1) and Jacobian (2 x 16+3N)
    def h_pixel_vel(self, x, i, u):
        # assert x.shape == (xZ + 5 * self.len_features, 1) and isinstance(i, int) and u.shape == (6, 1)

        vel = x[xVEL:xVEL + 3]
        omega = u[uG:uG+3] - x[xB_G:xB_G+3]

        # Camera Dynamics
        vel_c_i = self.q_b_c.invrot(vel + skew(omega).dot(self.p_b_c))
        omega_c_i = self.q_b_c.invrot(omega)

        q_c_z = Quaternion(x[xZ+i*5:xZ+i*5+4])
        rho = x[xZ+i*5+4]
        zeta = q_c_z.rot(self.khat)

        sk_vel = skew(vel_c_i)
        sk_ez = skew(self.khat)
        sk_zeta = skew(zeta)
        R_b_c = self.q_b_c.R

        # TODO: Need to convert to camera dynamics

        h = -self.focal_len*I_2x3.dot(sk_ez).dot(rho*(sk_zeta.dot(vel_c_i)) + omega_c_i)

        ZETA_i = dxZ+3*i
        RHO_i = dxZ+3*i+2
        dhdx = np.zeros((2,dxZ+3*self.len_features))
        dhdx[:,dxVEL:dxVEL+3] = -self.focal_len*rho*I_2x3.dot(sk_ez).dot(sk_zeta)
        dhdx[:,ZETA_i:ZETA_i+2] = -self.focal_len*rho*I_2x3.dot(sk_ez).dot(sk_vel).dot(sk_zeta).dot(T_zeta(q_c_z))
        dhdx[:,RHO_i,None] = -self.focal_len*I_2x3.dot(sk_ez).dot(sk_zeta).dot(vel_c_i)
        dhdx[:,dxB_G:dxB_G+3] = self.focal_len*I_2x3.dot(sk_ez).dot(R_b_c - rho*sk_zeta.dot(R_b_c).dot(skew(self.p_b_c)))
        # dhdx[:, dxB_G:dxB_G + 3] = self.focal_len * I_2x3.dot(sk_ez).dot(I_zz)


        return h, dhdx




