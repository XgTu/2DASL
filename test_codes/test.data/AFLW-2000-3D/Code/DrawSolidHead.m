function DrawSolidHead(vertex, tri, keyPoints)

trisurf(tri', vertex(1, :), vertex(2, :), vertex(3, :), 0, 'edgecolor', 'none');

if nargin == 3
    keyPoints = double(keyPoints);
    hold on
    plot3(keyPoints(1, :), keyPoints(2, :), keyPoints(3, :), 'b.', 'MarkerSize', 17);
    
    %plot3(keyPoints(1, [1:8]), keyPoints(2, [1:8]), keyPoints(3, [1:8]) + 10, 'r.', 'MarkerSize', 13);
    %plot3(keyPoints(1, [10:end]), keyPoints(2, [10:end]), keyPoints(3, [10:end]) + 1, 'b.', 'MarkerSize', 11);
    
%     plot3(keyPoints(1, [10:17]), keyPoints(2, [10:17]), keyPoints(3, [10:17]) + 10, 'm.', 'MarkerSize', 13);
%     plot3(keyPoints(1, [1:9,18:end]), keyPoints(2, [1:9,18:end]), keyPoints(3, [1:9,18:end]) + 1, 'b.', 'MarkerSize', 11);

%     plot3(keyPoints(1, [1:8]), keyPoints(2, [1:8]), keyPoints(3, [1:8]) + 10, 'r.', 'MarkerSize', 13);
%     plot3(keyPoints(1, [9:16]), keyPoints(2, [9:16]), keyPoints(3, [9:16]) + 10, 'm.', 'MarkerSize', 13);
%     plot3(keyPoints(1, [17:end]), keyPoints(2, [17:end]), keyPoints(3, [17:end]) + 1, 'b.', 'MarkerSize', 11);
    
    msg = cell(1, size(keyPoints, 2));
    for i = 1 : size(keyPoints, 2)
        msg{i} = sprintf('%d', i);
    end
    text(keyPoints(1, :), keyPoints(2, :), keyPoints(3, :)+100, msg);
end

light('Position', [0 0 1], 'Style', 'infinite');
lighting gouraud
axis equal
axis vis3d
title('3D shape');
xlabel('x');
ylabel('y');
zlabel('z');
view([0 90]);