import numpy as np
from pyquat import Quaternion
import scipy.linalg
from tqdm import tqdm
from math_helper import norm, q_array_from_two_unit_vectors

def add_landmark(truth, landmarks, p_b_c, q_b_c, F, lambda_0):
    assert truth.shape[1] > 7 and landmarks.shape[1] == 3

    ids = []
    depths = []
    lambdas = []
    q_zetas = []
    time = truth[:,0]

    khat = np.array([[0, 0, 1.]]).T
    all_ids = np.array([i for i in range(len(landmarks))])

    for i in range(len(truth)):
        q = Quaternion(truth[i, 4:8, None])
        delta_pose = landmarks[:,:3] - (truth[i, 1:4] + q.invrot(p_b_c).T)
        dist = norm(delta_pose, axis=1)
        q = Quaternion(truth[i,4:8,None])
        zetas = q_b_c.invrot(q.invrot((delta_pose/dist[:,None]).T))

        # Remove Features Behind the camera
        valid_ids = all_ids[zetas[2,:] > 0.2]
        frame_lambdas = []
        frame_depths = []
        frame_ids = []
        frame_qzetas = []
        for id in valid_ids:
            # Simulate pixel measurements
            l = 1.0 / khat.T.dot(zetas[:,id]) * F.dot(zetas[:,id,None]) + lambda_0
            if 0 < l[0,0] < 640 and 0 < l[1,0] < 480:
                frame_lambdas.append(l[:,0])
                frame_depths.append(dist[id])
                frame_ids.append(id)
                frame_qzetas.append(Quaternion.from_two_unit_vectors(khat, zetas[:,id,None]).elements[:,0])
        frame_lambdas = np.array(frame_lambdas)
        frame_depths = np.array(frame_depths)
        frame_ids = np.array(frame_ids)
        frame_qzetas = np.array(frame_qzetas)

        ids.append(frame_ids)
        depths.append(frame_depths)
        lambdas.append(frame_lambdas)
        q_zetas.append(frame_qzetas)

    return time, ids, lambdas, depths, q_zetas

def test():
    landmarks = np.random.uniform(-100, 100, (3,10))
    truth = []
    position = np.zeros((3,1))
    orientation = Quaternion.Identity()
    for i in range(1000):
        position += np.random.normal(0.0, 0.025, (3,1))
        orientation += np.random.normal(0.0, 0.025, (3,1))

        truth.append(np.hstack(np.array([[i]]),
                     position.T,
                     orientation.elements.T))
    truth = np.array(truth).squeeze()

    feature_time, zetas, depths, ids = add_landmark(truth, landmarks)





