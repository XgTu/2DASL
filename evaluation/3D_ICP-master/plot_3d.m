function plot_3d( data, theta )

    data = rotate(data, theta);
    x = data(:, 1);
    y = data(:, 2);
    z = data(:, 3);
    % plot3(x, y, z, 'k');
    figure();
    scatter3(x, y, z, 'k');

end