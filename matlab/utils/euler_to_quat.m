function quat = euler_to_quat(euler)
% uses the flight dynamics convention: roll about x -> pitch about y -> yaw
% about z (http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles)
c_1 = cos(euler(1)*0.5);
s_1 = sin(euler(1)*0.5);
c_2 = cos(euler(2)*0.5);
s_2 = sin(euler(2)*0.5);
c_3 = cos(euler(3)*0.5);
s_3 = sin(euler(3)*0.5);

quat(1) = c_1*c_2*c_3 + s_1*s_2*s_3;
quat(2) = s_1*c_2*c_3 - c_1*s_2*s_3;
quat(3) = c_1*s_2*c_3 + s_1*c_2*s_3;
quat(4) = c_1*c_2*s_3 - s_1*s_2*c_3;

end