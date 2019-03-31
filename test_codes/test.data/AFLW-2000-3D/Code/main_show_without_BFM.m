%% Load Model
data_path = '../';

%% Load Sample
sample_name = 'image00002';
img = imread([data_path sample_name '.jpg']);
load([data_path sample_name '.mat']);
[height, width, nChannels] = size(img);

imshow(img);
hold on;
plot(pt3d_68(1,:), pt3d_68(2,:), 'b.');
hold off;


