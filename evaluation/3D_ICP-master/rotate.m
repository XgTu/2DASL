function new_data = rotate( data, theta )
% 将原始点集data向上旋转theta角度，返回new_data。
% data：原始点集，每行都是一个点的x、y、z坐标；
% theta：旋转的角度，绕x轴顺时针旋转为正，即向下旋转为正；
% new_data：旋转后的点集。

    theta = - theta * pi / 180; % 角度转为弧度，取负值，表示向上旋转
    % 旋转矩阵
    matrix_rotate = [1, 0, 0, 0; 
                     0, cos(theta), sin(theta), 0;
                     0, -sin(theta), cos(theta), 0;
                     0, 0, 0, 1];
    rows = size(data, 1);   % 建立一个与data点集对应的矩阵
    row_ones = ones(rows, 1);   % 补1，将data点集扩展成齐次形式
    new_data = [data, row_ones] * matrix_rotate;    % 乘以旋转矩阵
    new_data = new_data(:, 1:3);    % 从齐次形式还原

end

