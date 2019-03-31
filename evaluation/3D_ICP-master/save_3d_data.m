function save_3d_data( path, data )
% 保存最终摆正之后的三维点云

    f = fopen(path, 'w');
    [m, n] = size(data);
    for i = 1:m
        for j = 1:n
            if j == n
                fprintf(f, '%g\r\n', data(i, j));
            else
                fprintf(f, '%g ', data(i, j));
            end
            
        end
    end
    fclose(f);

end

