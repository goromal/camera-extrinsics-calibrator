function [phi, theta, psi] = quat_to_euler_vecs(qw, qx, qy, qz)
% uses the flight dynamics convention: roll about x -> pitch about y -> yaw
% about z (http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles)
% WRITTEN FOR ELEMENT-WISE OPERATION ON VECTORS, RETURNING VECTORS

yr = 2 * (qw .* qx + qy .* qz);
xr = 1 - 2 * (qx.^2 + qy.^2);
phi = atan2(yr, xr);

sp = 2 * (qw .* qy - qz .* qx);
theta = asin(sp);

yy = 2 * (qw .* qz + qx .* qy);
xy = 1 - 2 * (qy.^2 + qz.^2);
psi = atan2(yy, xy);

end