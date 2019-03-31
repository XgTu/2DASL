function DrawTextureHead(vertex, tri, color, keyPoints)
n = size(vertex, 2);

% draw
trisurf(tri', vertex(1, :), vertex(2, :), vertex(3, :), 1 : n, 'edgecolor', 'none');

if nargin == 4
    hold on
    plot3(keyPoints(1, :), keyPoints(2, :), keyPoints(3, :), 'g.', 'MarkerSize', 20)
end

% color
colormap(color');

shading interp
axis equal
%axis([1 250 1 250])
axis vis3d
title('fitted head');
xlabel('x');
ylabel('y');
zlabel('z');
view([0,90]);