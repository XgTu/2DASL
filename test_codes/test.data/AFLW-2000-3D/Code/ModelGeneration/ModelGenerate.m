load('01_MorphableModel.mat');
load('model_info.mat');
trimIndex1 = [3*trimIndex-2, 3*trimIndex-1, 3*trimIndex]';
trimIndex1 = trimIndex1(:);

mu_shape = shapeMU(trimIndex1);
w = shapePC(trimIndex1, :);
sigma = shapeEV;

mu_tex = texMU(trimIndex1);
w_tex = texPC(trimIndex1,:);

segbin = segbin(trimIndex, :);

save('Model_Shape.mat', 'mu_shape', 'w', 'sigma', 'mu_tex', 'w_tex', 'tri', 'keypoints', 'symlist', 'symlist_tri', 'segbin', 'segbin_tri');


