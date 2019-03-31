function N = NormDirection(vertex, tri)

% norm of each triangles
pt1 = vertex(:, tri(1, :));
pt2 = vertex(:, tri(2, :));
pt3 = vertex(:, tri(3, :));
n_tri = cross(pt1 - pt2, pt1 - pt3);

% norm of each vertex
N = zeros(3, size(vertex, 2));
% for i = 1 : size(tri, 2)
%     N(:, tri(1, i)) = N(:, tri(1, i)) + n_tri(:, i);
%     N(:, tri(2, i)) = N(:, tri(2, i)) + n_tri(:, i);
%     N(:, tri(3, i)) = N(:, tri(3, i)) + n_tri(:, i);
% end

N = Tnorm_VnormC(double(n_tri), double(tri), double(size(tri,2)), double(size(vertex,2)));

% normalize to unit length
mag = sum(N .* N);
% deal with zero vector
co = find(mag == 0);
mag(co) = 1;
N(1, co) = ones(length(co),1);
N = N ./ sqrt(repmat(mag, 3, 1));
N = -N;