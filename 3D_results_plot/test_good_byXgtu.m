clear all; close all;
addpath(genpath('./visualize'));
tri = load('tri.mat');
aa = dir('./aflw-2000_shown/2DASL_vertices/*.mat');
tri = tri.tri;

for ii = 1:length(aa)
    img = imread(strcat('./aflw-2000_shown/oriImgs/', aa(ii).name(1:end-4), '.jpg'));
    vertex = load(strcat('./aflw-2000_shown/2DASL_vertices/', aa(ii).name(1:end-4)));
    figure
    imshow(img)
    im1 = imagesc(img); 
    hold on
    
    vertex = vertex.vertex;
    vertex(3,:,:) = vertex(3,:,:) - min(vertex(3,:,:));
    pcshow(vertex')
    view(2)
    saveas(gca, strcat('./aflw-2000_shown/results/', 'dense_align_', aa(ii).name(1:end-4), '.jpg'))
    close all
    figure
    im1 = imagesc(img); 
    hold on
    render_face_mesh_xgtu(vertex, tri);
    saveas(gca, strcat('./aflw-2000_shown/results/', 'recons_align_', aa(ii).name(1:end-4), '.jpg'))
    close all
end

print('done')