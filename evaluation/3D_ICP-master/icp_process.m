function [ data_g, data_p, err, data_pp, R ] = icp_process( data_g, data_p )
% 对两个点集data_g和data_p应用ICP算法
% data_g\data_P:样本点集
% 返回旋转后的两个点集，以及误差均值以及data_p对应data_g的对应点集data_pp

    [k1, n] = size(data_g);
    [k2, m] = size(data_p);
    
    data_p1 = zeros(k2, 3);     % 中间点集
    data_pp = zeros(k1, 3);     % 对应点集
    distance = zeros(k1, 1);    % 点集之间各点的距离
    error = zeros(k1, 1);       % 对应点之间的误差
    
    % 对两个点集作去中心化
    data_g = normal_gravity(data_g);
    data_p = normal_gravity(data_p);
    
    % 遍历整个data_g点集，寻找每个点对应data_p点集中距离最小的点，作为对应点
    for i = 1:k1
        data_p1(:, 1) = data_p(:, 1) - data_g(i, 1);    % 两个点集中的点x坐标之差
        data_p1(:, 2) = data_p(:, 2) - data_g(i, 2);    % 两个点集中的点y坐标之差
        data_p1(:, 3) = data_p(:, 3) - data_g(i, 3);    % 两个点集中的点z坐标之差
        distance = data_p1(:, 1).^2 + data_p1(:, 2).^2 + data_p1(:, 3).^2;  % 欧氏距离
        [min_dis, min_index] = min(distance);   % 找到距离最小的那个点
        data_pp(i, :) = data_p(min_index, :);   % 将那个点保存为对应点
        error(i) = min_dis;     % 保存距离差值
    end

    % 求出协方差矩阵
    V = (data_g' * data_pp) ./ k1;
    
    % 构建正定矩阵Q（这部分不是很理解，直接套公式了）
    matrix_Q = [V(1,1)+V(2,2)+V(3,3),V(2,3)-V(3,2),V(3,1)-V(1,3),V(1,2)-V(2,1);  
                V(2,3)-V(3,2),V(1,1)-V(2,2)-V(3,3),V(1,2)+V(2,1),V(1,3)+V(3,1);  
                V(3,1)-V(1,3),V(1,2)+V(2,1),V(2,2)-V(1,1)-V(3,3),V(2,3)+V(3,2);  
                V(1,2)-V(2,1),V(1,3)+V(3,1),V(2,3)+V(3,2),V(3,3)-V(1,1)-V(2,2)];
    
    [V2, D2] = eig(matrix_Q);       % 对矩阵Q作特征值分解
    lambdas = [D2(1, 1), D2(2, 2), D2(3, 3), D2(4, 4)]; % 取出特征值
    [lambda, ind] = max(lambdas);   % 求出最大的那个特征值
    Q = V2(:, ind); % 取出那个最大的特征值所对应的特征向量
    
    % 构建旋转矩阵（四元数）
    R=[Q(1,1)^2+Q(2,1)^2-Q(3,1)^2-Q(4,1)^2,     2*(Q(2,1)*Q(3,1)-Q(1,1)*Q(4,1)),        2*(Q(2,1)*Q(4,1)+Q(1,1)*Q(3,1));  
       2*(Q(2,1)*Q(3,1)+Q(1,1)*Q(4,1)),         Q(1,1)^2-Q(2,1)^2+Q(3,1)^2-Q(4,1)^2,    2*(Q(3,1)*Q(4,1)-Q(1,1)*Q(2,1));  
       2*(Q(2,1)*Q(4,1)-Q(1,1)*Q(3,1)),         2*(Q(3,1)*Q(4,1)+Q(1,1)*Q(2,1)),        Q(1,1)^2-Q(2,1)^2-Q(3,1)^2-Q(4,1)^2;  
    ];
    
    % 对data_p点集所有的点都做R的旋转变化，然后再作中心平移
    data_p = data_p * R;
    data_pp = data_pp * R;
    data_p = normal_gravity(data_p);
    data_pp = normal_gravity(data_pp);
    err = mean(error);
    
end

