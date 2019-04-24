#!/usr/bin/env python


from pyquat import Quaternion
import numpy as np
from math_helper import norm
import sys


def print_array_yaml(name, arr):
	sys.stdout.write(name + ': [')
	for row in range(arr.shape[0]):
		for col in range(arr.shape[1]):
			sys.stdout.write("%f" % arr[row,col])
			if col == arr.shape[1] - 1:
				if row == arr.shape[0] - 1:
					sys.stdout.write('],')
				else:
					sys.stdout.write(',')
			else:
				sys.stdout.write(', ')				
		sys.stdout.write('\n')

T_IMU_c = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
         			[ 0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
         			[-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
         			[ 0.0, 0.0, 0.0, 1.0]])
R_IMU_c = T_IMU_c[:3,:3]
p_b_c_IMU = T_IMU_c[:3,3,None]

T_IMU_vicon = np.array([[ 0.33638, -0.01749,  0.94156,  0.06901],
         	  			[-0.02078, -0.99972, -0.01114, -0.02781],
         	  			[ 0.94150, -0.01582, -0.33665, -0.12395],
              			[ 0.0,      0.0,      0.0,      1.0]])
R_IMU_vicon = T_IMU_vicon[:3,:3]

# R_vicon_b = np.array([[1, 0, 0],
# 					  [0, -1, 0],
# 					  [0, 0, -1]]).dot(np.array([[-1, 0, 0],[0, 1, 0], [0, 0, -1]]))
q_vicon_b = Quaternion(np.array([[0, 0, 0, -1.0]]).T)
R_vicon_b = np.array([[1, 0, 0],
					  [0, -1, 0],
					  [0, 0, -1]])
q_vicon_b = Quaternion(np.array([[0, 0, 0, -1.0]]).T) * Quaternion(np.array([[0, 1.0, 0, 0]]).T)
q0 = Quaternion(np.array([[0, 0, 0, 1]]).T)
print ("q0 = ", q_vicon_b * q0)
print ("####################################")

R_b_IMU =(R_vicon_b.dot(R_IMU_vicon))


R_b_c = R_b_IMU.dot(R_IMU_c)
p_b_c_b = R_b_IMU.T.dot(p_b_c_IMU)



b_w_0 = np.array([[-0.002295, 0.024939, 0.081667]]).T
b_a_0 = np.array([[-0.023601, 0.121044, 0.074783]]).T

b_w_f = np.array([[-0.002275, 0.024883, 0.081577]]).T
b_a_f = np.array([[-0.019635, 0.123542, 0.07849]]).T

w_f = np.array([[-0.0055850536, 0.0244346095, 0.0830776724]]).T
a_f = np.array([[9.3571785417, 0.2043052083, -2.819411875]]).T

q_b_IMU = Quaternion.from_R(R_b_IMU)

# calculate orientation of IMU at final
# a = a_f - b_a_f
# a /= norm(a)
# g = np.array([[0, 0, -1]]).T
# q_b_IMU = Quaternion.from_two_unit_vectors(g, a)
# R_b_IMU = q_b_IMU.R

print ("norm a_f", norm(a_f-b_a_f))
print ("norm w_f", norm(w_f-b_w_f))
print_array_yaml("q_b_IMU", q_b_IMU.elements.T)
print("######################################")
# q_IMU_b = Quaternion.from_two_vectors()




print_array_yaml("q_b_IMU", q_b_IMU.elements.T)
print_array_yaml("q_b_c", Quaternion.from_R(R_b_c).elements.T)
print_array_yaml("p_b_c_b", p_b_c_b.T)
print_array_yaml("b_w_0", -R_b_IMU.T.dot(b_w_0).T)
print_array_yaml("b_a_0", -R_b_IMU.T.dot(b_a_0).T)
print_array_yaml("q_I_truth", q_vicon_b.elements.T)






