% aa = dir('../3D_results_plot/results/3dLandmarks_proj_resnet50/*.mat');
clear all;
close all;
grdPath = './results/AFLW-2000-3D_grdth/';
grdDir = dir('./results/AFLW-2000-3D_grdth/*.jpg');
bb = dir('./results/data/DeFA/mesh/*.mat');
% cc = dir('../3D_results_plot/results/AFLW-2000-3D/*.mat');
aa = textread('./keypoints.txt', '%s');
failDect = textread('./results/data/fail_detected.txt', '%s');
yaws = load('0-30-60-90.mat');
zero_thrity = cellstr(yaws.zero_thirty);
thrity_sixty = cellstr(yaws.thirty_sixty);
sixty_ninty = cellstr(yaws.sixty_ninty);

pts = zeros(204, 1);
for i = 1:204
    xx = aa(i);
    xx = xx{1};
    pts(i) = str2num(xx(1:end-1));
end
pts = reshape(pts, 3, 68) + 1;
index = pts(3,:)/3;

nme_list_2DASSL = zeros(2, 1000);
nme_list_3DDFA = zeros(2, 1000);
ze_th = [];
th_si = [];
si_ni = [];
for ii = 1:2000
    ii
    imgName = grdDir(ii).name;
    if (ismember(imgName, failDect))
        continue
    end
    info = load(strcat(grdPath,strrep(imgName, 'jpg', 'mat')));
    pts1 = info.pt3d_68;
    pts1 = pts1(1:2,:);
    img = imread(strcat(grdPath, imgName));
    
    mmat = load(strcat('./results/data/DeFA/mesh/', imgName(1:end-4), '_mesh.mat'));
    vertex = mmat.mesh.vertices;
    pts2 = vertex(:,index);
    pts2 = pts2(1:2,:);
%     pp = load(strcat('./results/data/DeFA/kpt/', imgName(1:end-4), '_kpt.txt'));
%     pts2 = pp(:,1:2)';
    
    pp = textread(strcat('./results/data/PRNet/kpt/', imgName(1:end-4), '_kpt.txt'));
    pts3 = pp(:,1:2)';
    
    pp = textread(strcat('./results/data/vrn/ktp/', imgName(1:end-4), '_kpt.txt'));
    pts4 = pp(:,1:2)';
    
    vertex = load(strcat('./results/3dLandmarks_proj_resnet50_2DASSL_3_613/', strrep(imgName, 'jpg', 'mat')));
    pts5 = vertex.vertex(:,index);
    pts5 = pts5(1:2,:);
   
    vertex = load(strcat('./results/3DDFA_results/AFLW-2000-3D_vertex_pdc/', strrep(imgName, 'jpg', 'mat')));
    pts6 = vertex.vertex(:,index);
    pts6 = pts6(1:2,:);
    
%     imshow(img);
%     hold on
%     for i = 1:68
%         plot(pts2(1,i), pts2(2,i), 'o');
%     end
    
    % coords = 37/46 ���۽Ǽ����
    minx = min(pts1(1,:)); miny = min(pts1(2,:));
    maxx = max(pts1(1,:)); maxy = max(pts1(2,:));
    llen = sqrt((maxx-minx)*(maxy-miny));
    dis1 = (pts1 - pts2);
    dis1 = sqrt(sum(dis1.^2));
    dis1 = mean(dis1)/llen;
    
    dis2 = abs(pts1 - pts3);
    dis2 = sqrt(sum(dis2.^2));
    dis2 = mean(dis2)/llen;
    
    dis3 = abs(pts1 - pts4);
    dis3 = sqrt(sum(dis3.^2));
    dis3 = mean(dis3)/llen;
    
    dis4 = abs(pts1 - pts5);
    dis4 = sqrt(sum(dis4.^2));
    dis4 = mean(dis4)/llen;
    
    if (ismember(imgName, zero_thrity))
        ze_th = [ze_th dis4];

    end
    
    if (ismember(imgName, thrity_sixty))
        th_si = [th_si dis4];

    end
    
    if (ismember(imgName, sixty_ninty))
        si_ni = [si_ni dis4];

    end
    
    dis5 = abs(pts1 - pts6);
    dis5 = sqrt(sum(dis5.^2));
    dis5 = mean(dis5)/llen;

    nme_list_DeFA(1, ii) = dis1;
    nme_list_PRNet(1, ii) = dis2;
    nme_list_vrn(1, ii) = dis3;
    nme_list_2DASSL(1, ii) = dis4;
    nme_list_3DDFA(1, ii) = dis5;
    
%     roi = info.roi;
%     ll = eyeDis;
end

% save('nme_list', 'nme_list');
LL = 1980;
dis_DeFA = nme_list_DeFA(1,:);
dis_PRNet = nme_list_PRNet(1,:);
dis_vrn = nme_list_vrn(1,:);
dis_2DASSL = nme_list_2DASSL(1,:);
dis_3DDFA = nme_list_3DDFA(1,:);

[s_dis_DeFA, index] = sort(dis_DeFA);
[s_dis_PRNet, index] = sort(dis_PRNet);
[s_dis_vrn, index] = sort(dis_vrn);
[s_dis_2DASSL, index] = sort(dis_2DASSL);
[s_dis_3DDFA, index] = sort(dis_3DDFA);

s_dis_DeFA = s_dis_DeFA(30:end);
s_dis_PRNet = s_dis_PRNet(30:end);
s_dis_vrn = s_dis_vrn(30:end);
s_dis_3DDFA = s_dis_3DDFA(30:end);
s_dis_2DASSL = s_dis_2DASSL(30:end);

% s_dis_2DASSL = s_dis_2DASSL(1:(length(s_dis_2DASSL)/100):length(s_dis_2DASSL));
% s_dis_3DDFA = s_dis_3DDFA(1:(length(s_dis_3DDFA)/100):length(s_dis_3DDFA));

% s_dis_2DASSL = s_dis_2DASSL(1:(length(s_dis_2DASSL)/100):length(s_dis_2DASSL));
% s_dis_3DDFA = s_dis_3DDFA(1:(length(s_dis_3DDFA)/100):length(s_dis_3DDFA));
% minNum = min(min(s_dis_2DASSL), min(s_dis_3DDFA));
% s_dis_2DASSL = (s_dis_2DASSL - minNum);
% s_dis_3DDFA = (s_dis_3DDFA - minNum);
% maxNum = min(max(s_dis_2DASSL), max(s_dis_3DDFA));
% s_dis_2DASSL = s_dis_2DASSL/maxNum;
% s_dis_3DDFA = s_dis_3DDFA/maxNum;

x_len = 0:1:length(s_dis_DeFA);
x_len = x_len(1:length(s_dis_DeFA));
plot(s_dis_DeFA*100, x_len, 'b', 'linewidth',2);
hold on
plot(s_dis_PRNet*100, x_len, 'c', 'linewidth',2);
hold on
plot(s_dis_vrn*100, x_len, 'y', 'linewidth',2);
hold on
plot(s_dis_3DDFA*100, x_len, 'r', 'linewidth',2);
hold on
plot(s_dis_2DASSL*100, x_len, 'g', 'linewidth',2);

axis([0 10 0 2000]) 
set(gca,'XLim',[0 8]);%X���������ʾ��Χ
set(gca,'YLim',[0 2000]);%X���������ʾ��Χ

grid on
grid minor

ax = gca;
ax.GridColor = [0 .5 .5];
ax.GridLineStyle = '--';
ax.GridAlpha = 0.5;
ax.Layer = 'top';

h = legend('DeFA: 4.79', 'PRNet: 3.63', 'VRN: 3.86', '3DDFA: 4.55', '2DASSL: 3.16', 'Location','southeast')
%set(h,'Orientation','horizon', 'Fontsize',12)
set(h,'Fontsize',12)

xlabel('NME normalized by bounding box', 'fontsize', 12)
ylabel('Number of images', 'fontsize',12)