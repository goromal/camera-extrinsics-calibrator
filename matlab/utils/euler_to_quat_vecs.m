function [qw, qx, qy, qz] = euler_to_quat_vecs(phi, theta, psi)
% uses the flight dynamics convention: roll about x -> pitch about y -> yaw
% about z (http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles)
% WRITTEN FOR ELEMENT-WISE OPERATION ON VECTORS, RETURNING VECTORS

c_1 = cos(phi*0.5);
s_1 = sin(phi*0.5);
c_2 = cos(theta*0.5);
s_2 = sin(theta*0.5);
c_3 = cos(psi*0.5);
s_3 = sin(psi*0.5);

qw = c_1.*c_2.*c_3 + s_1.*s_2.*s_3;
qx = s_1.*c_2.*c_3 - c_1.*s_2.*s_3;
qy = c_1.*s_2.*c_3 + s_1.*c_2.*s_3;
qz = c_1.*c_2.*s_3 - s_1.*s_2.*c_3;

end