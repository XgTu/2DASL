function ret = normal_gravity( data )
% 将点集data去中心化
% 其中行为不同样本，列为一个样本的特征

    [m, n] = size(data);
    data_mean = mean(data);
    ret = data - ones(m, 1) * data_mean;

end

