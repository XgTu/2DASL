function plot_3d_2( data1, data2, theta )

    data1 = rotate(data1, theta);
    x1 = data1(:, 1);
    y1 = data1(:, 2);
    z1 = data1(:, 3);

    data2 = rotate(data2, theta);
    x2 = data2(:, 1);
    y2 = data2(:, 2);
    z2 = data2(:, 3);
    
    figure();
    scatter3(x1, y1, z1, 'b');
    hold on;
    scatter3(x2, y2, z2, 'r');
    hold off;

end