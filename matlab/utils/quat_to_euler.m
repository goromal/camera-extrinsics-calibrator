function eulerXYZ = quat_to_euler(q)
% uses the flight dynamics convention: roll about x -> pitch about y -> yaw
% about z (http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles)
qw = q(1);
qx = q(2);
qy = q(3);
qz = q(4);

yr = 2*(qw*qx + qy*qz);
xr = 1 - 2*(qx^2 + qy^2);
roll = atan2(yr, xr);

sp = 2*(qw*qy - qz*qx);
pitch = asin(sp);

yy = 2*(qw*qz + qx*qy);
xy = 1 - 2*(qy^2 + qz^2);
yaw = atan2(yy, xy);

eulerXYZ = [roll pitch yaw];

end