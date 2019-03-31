% This script is used to evluation dense face alignment performance (2d coordinates) using NME metric, 
% The script contains the process of 3D face mesh reconstruction from the
% 3DMM parameters
clear all;
close all;
addpath(genpath('./visualize'));
aa = textread('./keypoints.txt', '%s');
bb = dir('./results/AFLW-2000-3D_grdth/*.mat');
load('Model_Expression.mat');
load('Model_Shape.mat');
load('BaseSample.mat');

std_size = 120;
vertex_mean = reshape(mu_shape, 3, length(mu_shape)/3);
mu_id = mu_shape + mu_exp;

base_ind = keypoints;
base_ind1 = [(3 * base_ind - 2); (3 * base_ind -1); (3 * base_ind)]; 
base_ind1 = base_ind1(:);
mu_base = mu_id(base_ind1);
w_base = w_shape(base_ind1,:);
w_exp_base = w_exp(base_ind1,:);

pts = zeros(204, 1);
for i = 1:204
    xx = aa(i);
    xx = xx{1};
    pts(i) = str2num(xx(1:end-1));
end
pts = reshape(pts, 3, 68) + 1;
index = pts(3,:)/3;

LL = 40;
nme_list_2DASSL = zeros(2, LL);
nme_list_3DDFA = zeros(2, LL);
for ii = 1:LL
    ii
    imgName = bb(ii).name;
    img = imread(strcat('./results/AFLW-2000-3D_grdth/', strrep(imgName,'mat', 'jpg')));
    
    [height, width, nChannels] = size(img);
    info = load(strcat('./results/AFLW-2000-3D_grdth/', imgName));
    pt3d_68 = info.pt3d_68;
    Pose_Para = info.Pose_Para;
    pts1 = pt3d_68;
    
    % Reconstruct 3D point of base point
    project_base_point = pt3d_68;
    bbox = [min(project_base_point(1,:)), min(project_base_point(2,:)), max(project_base_point(1,:)), max(project_base_point(2,:))];
    
    center = [(bbox(1)+bbox(3))/2, (bbox(2)+bbox(4))/2];
    radius = max(bbox(3)-bbox(1), bbox(4)-bbox(2)) / 2;
    bbox = [center(1) - radius, center(2) - radius, center(1) + radius, center(2) + radius];
    

    bbox = double(bbox);
    widthb = vertex_mean(1,keypoints(17)) - vertex_mean(1,keypoints(1));
    heightb = vertex_mean(2, keypoints(20)) - vertex_mean(2, keypoints(9));

    mean_x = vertex_mean(1, keypoints(31));
    mean_y = vertex_mean(2, keypoints(31));

    f0 = ((bbox(3) - bbox(1)) / widthb + (bbox(4) - bbox(2)) / heightb) / 2;
    t3d0(1) = (bbox(3) + bbox(1))/2 - f0*mean_x;
    temp = height + 1 - (bbox(2) + bbox(4))/2;
    t3d0(2) = temp - f0 * mean_y;
    t3d0(3) = 0;

    Pose_Para0 = [Pose_Para(1:3), t3d0(1), t3d0(2), t3d0(3), f0];
%    Shape_Para0 = zeros(size(Shape_Para));
%    Exp_Para0 = zeros(size(Exp_Para));
    Shape_Para = info.Shape_Para;
    Tex_Para = info.Tex_Para;
    Pose_Para = info.Pose_Para;

    [phi, gamma, theta, t3d, f] = ParaMap_Pose(Pose_Para);
    R = RotationMatrix(phi, gamma, theta);

    alpha_shape = Shape_Para;
    alpha_tex = Tex_Para;
    alpha_exp = Exp_Para;
    express = w_exp * alpha_exp; express = reshape(express, 3, length(express)/3);
    shape = mu_shape + w_shape * alpha_shape; shape = reshape(shape, 3, length(shape)/3);
    tex = mu_tex + w_tex * alpha_tex; tex = reshape(tex, 3, length(tex)/3);
    vertex = shape + express;
    grdVertex = f * R * vertex + repmat(t3d, 1, size(vertex, 2));
    
%     figure
%     imshow(uint8(img), [])
%     hold on
    
    grdVertex(3,:,:) = grdVertex(3,:,:) - min(grdVertex(3,:,:));
    grdVertex(2,:,:) = 1-grdVertex(2,:,:)+450;
%     pcshow(grdVertex')
%     view(2)

    minx = min(pts1(1,:)); miny = min(pts1(2,:));
    maxx = max(pts1(1,:)); maxy = max(pts1(2,:));
    llen = sqrt((maxx-minx)*(maxy-miny));
    
    vertex1 = load(strcat('./results/2DASL_results/', imgName));
    vertex1 = vertex1.vertex;
    vertex1(3,:,:) =  vertex1(3,:,:) - min(vertex1(3,:,:));

    dis1 = grdVertex(1:2,:) - vertex1(1:2,:);
    dis1 = sqrt(sum(dis1.^2));
    dis1 = mean(dis1)/llen;
    
%     pcshow(vertex1')
%     view(2)

    vertex2 = load(strcat('./results/3DDFA_results/', imgName));
    vertex2 = vertex2.vertex;
    vertex2(3,:,:) =  vertex2(3,:,:) - min(vertex2(3,:,:));
    
    dis2 = grdVertex(1:2,:) - vertex2(1:2,:);
    dis2 = sqrt(sum(dis2.^2));
    dis2 = mean(dis2)/llen;
%     pcshow(vertex2')
%     view(2)

    nme_list_2DASSL(1, ii) = dis1;
    nme_list_3DDFA(1, ii) = dis2;

end

% save('nme_list', 'nme_list');
dis_2DASSL = nme_list_2DASSL(1,:);
dis_3DDFA = nme_list_3DDFA(1,:);

[s_dis_2DASSL, index] = sort(dis_2DASSL);
[s_dis_3DDFA, index] = sort(dis_3DDFA);

s_dis_2DASSL = s_dis_2DASSL(1:LL);
s_dis_3DDFA = s_dis_3DDFA(1:LL);

x_len = 0:1:length(s_dis_2DASSL);
x_len = x_len(1:length(s_dis_2DASSL));
plot(s_dis_3DDFA*100, x_len, 'r', 'linewidth',2);
hold on
plot(s_dis_2DASSL*100, x_len, 'g', 'linewidth',2);

axis([0 10 0 LL]) 
set(gca,'XLim',[0 10]);
set(gca,'YLim',[0 LL]);

grid on
grid minor

ax = gca;
ax.GridColor = [0 .5 .5];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.Layer = 'top';

h = legend('3DDFA', '2DASSL', 'Location','southeast')
%set(h,'Orientation','horizon', 'Fontsize',12)
set(h,'Fontsize',12)

xlabel('NME normalized by bounding box', 'fontsize', 12)
ylabel('Number of images', 'fontsize',12)