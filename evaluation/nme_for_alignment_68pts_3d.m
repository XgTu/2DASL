% aa = dir('../3D_results_plot/results/3dLandmarks_proj_resnet50/*.mat');
clear all;
close all;
grdPath = './results/AFLW-2000-3D_grdth/';
grdDir = dir('./results/AFLW-2000-3D_grdth/*.jpg');
aa = textread('./keypoints.txt', '%s');
bb = dir('./results/data/DeFA/mesh/*.mat');

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
    imgName = grdDir(ii).name;
    info = load(strcat(grdPath,strrep(imgName, 'jpg', 'mat')));
    pts1 = info.pt3d_68;
    pts1 = pts1(1:3,:);
    pts1(3,:) = pts1(3,:) -min(pts1(3,:));
    img = imread(strcat(grdPath, imgName));
    
    pp = textread(strcat('./results/PRNet_results/kpt/', imgName(1:end-4), '_kpt.txt'));
    pts3 = pp(:,1:3)';
    pts3(3,:) = pts3(3,:) -min(pts3(3,:));
    
    vertex = load(strcat('./results/2DASL_results/', strrep(imgName, 'jpg', 'mat')));
    pts5 = vertex.vertex(:,index);
    pts5 = pts5(1:3,:);
    pts5(3,:) = pts5(3,:) -min(pts5(3,:));
    
    
%     imshow(img);
%     hold on
%     for i = 1:68
%         plot(pts1(1,i), pts1(2,i), 'o');
%     end
    
    % NME calculation
    minx = min(pts1(1,:)); miny = min(pts1(2,:));
    maxx = max(pts1(1,:)); maxy = max(pts1(2,:));
    llen = sqrt((maxx-minx)*(maxy-miny));
    
    dis2 = abs(pts1 - pts3);
    dis2 = sqrt(sum(dis2.^2));
    dis2 = mean(dis2)/llen;
    
    dis4 = abs(pts1 - pts5);
    dis4 = sqrt(sum(dis4.^2));
    dis4 = mean(dis4)/llen;
    
    nme_list_PRNet(1, ii) = dis2;
    nme_list_2DASSL(1, ii) = dis4;
end

% save('nme_list', 'nme_list');
dis_PRNet = nme_list_PRNet(1,:);
dis_2DASSL = nme_list_2DASSL(1,:);

[s_dis_PRNet, index] = sort(dis_PRNet);
[s_dis_2DASSL, index] = sort(dis_2DASSL);

s_dis_PRNet = s_dis_PRNet(1:LL);
s_dis_2DASSL = s_dis_2DASSL(1:LL);

x_len = 0:1:length(s_dis_PRNet);
x_len = x_len(1:length(s_dis_PRNet));
plot(s_dis_PRNet*100, x_len, 'c', 'linewidth',2);
hold on
plot(s_dis_2DASSL*100, x_len, 'g', 'linewidth',2);

axis([0 10 0 LL]) 
set(gca,'XLim',[0 10]);%
set(gca,'YLim',[0 LL]);%

grid on
grid minor

ax = gca;
ax.GridColor = [0 .5 .5];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.Layer = 'top';

h = legend('PRNet', '2DASSL', 'Location','southeast')
%set(h,'Orientation','horizon', 'Fontsize',12)
set(h,'Fontsize',12)

xlabel('NME normalized by bounding box', 'fontsize', 12)
ylabel('Number of images', 'fontsize',12)