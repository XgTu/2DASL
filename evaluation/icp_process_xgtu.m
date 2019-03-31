function [data_g,  data_p] = icp_process_xgtu(data_g,  data_p)

[ data_g, data_p, error, data_pp, R ] = icp_process(data_g, data_p);
log_info(strcat('迭代次数：1，误差：', num2str(error)));
log_info('当前旋转矩阵为：');
disp(R);

cnt = 1;
last_error = 0;
last_R = R;
% 当误差收敛时，停止循环
while abs(error - last_error) > 0.01
    cnt = cnt + 1;
    last_error = error;
    last_R = R;
    [data_g, data_p, error, data_pp, R] = icp_process(data_g, data_p);
    R = last_R * R;
    log_info(strcat('迭代次数：', num2str(cnt), '，误差：', num2str(error)));
    log_info('当前旋转矩阵为：');
    disp(R);
end
end

