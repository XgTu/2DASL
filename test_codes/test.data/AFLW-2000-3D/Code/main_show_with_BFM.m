%% Load Model
data_path = '../';
load('Model_Shape.mat');
load('Model_Exp.mat');
mu = mu_shape + mu_exp;

%% Load Sample
sample_name = 'image00004';
img = imread([data_path sample_name '.jpg']);
load([data_path sample_name '.mat']);
[height, width, nChannels] = size(img);

vertex = mu + w * Shape_Para + w_exp * Exp_Para;
vertex = reshape(vertex, 3, length(vertex)/3);
tex = mu_tex + w_tex * Tex_Para;
tex = reshape(tex, 3, length(mu_tex)/3);
norm = NormDirection(vertex, tri);

%% Pose Parameter
phi = Pose_Para(1); % pitch
gamma = Pose_Para(2); % yaw
theta = Pose_Para(3); % roll
t3dx = Pose_Para(4); % translation
t3dy = Pose_Para(5);
t3dz = Pose_Para(6);
f = Pose_Para(7); % scale

t3d = [t3dx; t3dy; t3dz];

R = RotationMatrix(phi, gamma, theta);
P = [1 0 0; 0 1 0];

%% Color Parameter
% Color and Illum Model can be refered to 
% Blanz et.al A morphable model for the synthesis of 3d faces, SIGGRAPH'99
Gain_r = Color_Para(1);
Gain_g = Color_Para(2);
Gain_b = Color_Para(3);

Offset_r = Color_Para(4);
Offset_g = Color_Para(5);
Offset_b = Color_Para(6);

c = Color_Para(7);

M = [0.3 0.59 0.11; 0.3 0.59 0.11; 0.3 0.59 0.11];
g = diag([Gain_r, Gain_g, Gain_b]);
o = [Offset_r; Offset_g; Offset_b];
o = repmat(o, 1, size(vertex,2));

%% Illumination Parameter
Amb_r = Illum_Para(1);
Amb_g = Illum_Para(2);
Amb_b = Illum_Para(3);
Dir_r = Illum_Para(4);
Dir_g = Illum_Para(5);
Dir_b = Illum_Para(6);
thetal = Illum_Para(7);
phil = Illum_Para(8);
ks = Illum_Para(9);
v = Illum_Para(10);

Amb = diag([Amb_r, Amb_g, Amb_b]);
Dir = diag([Dir_r, Dir_g, Dir_b]);
l = [cos(thetal)*sin(phil), sin(thetal), cos(thetal)*cos(phil)]';
h = l + [0,0,1]';
h = h / sqrt(h'*h);

%% Other paramters:
%roi: the crop region on the original image.
%pt2d: the original labelled landmarks.

%% Draw Fitted Face
ProjectVertex = f * R * vertex + repmat(t3d, 1, size(vertex, 2));

n_l = max(l' * norm,0);
n_h = max(h' * norm,0);
n_l = repmat(n_l, 3, 1);
n_h = repmat(n_h, 3, 1);
L = Amb * tex + Dir * (n_l .* tex) + ks * Dir * (n_h .^ v);
CT = g * (c * eye(3) + (1-c) * M);
tex_color = CT * L + o;
tex_color = double(tex_color)/255;
tex_color = min(max(tex_color,0),1);

imshow(img);
hold on;
pt3d = ProjectVertex(:, keypoints);
pt3d(2,:) = height + 1 - pt3d(2,:);
plot(pt3d(1,:), pt3d(2,:), 'b.');
hold off;
figure;
DrawTextureHead(ProjectVertex, tri, tex_color);
