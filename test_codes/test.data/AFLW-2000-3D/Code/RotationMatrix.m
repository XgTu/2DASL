function [R] = RotationMatrix(angle_x, angle_y, angle_z)
% get rotation matrix by rotate angle

phi = angle_x;
gamma = angle_y;
theta = angle_z;

R_x = [1 0 0 ; 0 cos(phi) sin(phi); 0 -sin(phi) cos(phi)];
R_y = [cos(gamma) 0 -sin(gamma); 0 1 0; sin(gamma) 0 cos(gamma)];
R_z = [cos(theta) sin(theta) 0; -sin(theta) cos(theta) 0; 0 0 1];

R = R_x * R_y * R_z;


end

