clear all; close all;
addpath(genpath('./visualize'));
tri = load('tri.mat');
% vertex = load('image00427');
% img = imread('image00427.jpg');
aa = dir('./aflw-2000_shown/oriImgs/*.jpg');
tri_2DASSL = tri.tri;

for ii = 1:length(aa)
    ii
    
    img = imread(strcat('./aflw-2000_shown/oriImgs/', aa(ii).name));
    
    mmat = load(strcat('./aflw-2000_shown/2DASL_vertices/', aa(ii).name(1:end-4), '.mat'));
    ver_2DASSL = mmat.vertex;
    write_obj(strcat('./', aa(ii).name(1:end-4), '.obj'), ver_2DASSL, tri_2DASSL);
    
end

print('done')