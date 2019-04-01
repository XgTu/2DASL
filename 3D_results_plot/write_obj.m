function write_obj(filename, vertex, face, options)
% write_off - write a mesh to an OBJ file
%   write_obj(filename, vertex, face, options)
%   vertex must be of size [n,3]
%   face must be of size [p,3]
%   Copyright (c) 2004 Gabriel Peyr¨¦

if nargin<4
    options.null = 0;
end

if size(vertex,2)~=3
    vertex=vertex';
end
if size(vertex,2)~=3
    error('vertex does not have the correct format.');
end


if size(face,2)~=3
    face=face';
end
if size(face,2)~=3
    error('face does not have the correct format.');
end

fid = fopen(filename,'wt');
if( fid==-1 )
    error('Can''t open the file.');
    return;
end

object_name = filename(1:end-4);

fprintf(fid, '# write_obj (c) 2004 Gabriel Peyr¨¦\n');
if isfield(options, 'nm_file')
    fprintf(fid, 'mtllib ./%s.mtl\n', object_name);
end

object_name = 'curobj';

fprintf(fid, ['g\n# object ' object_name ' to come\n']);

% vertex position
fprintf(fid, '# %d vertex\n', size(vertex,1));
fprintf(fid, 'v %f %f %f\n', vertex');

% vertex texture
if isfield(options, 'nm_file')
    nvert = size(vertex,1);
    object_texture = zeros(nvert, 2);
    m = ceil(sqrt(nvert));
    if m^2~=nvert
        error('To use normal map the number of vertex must be a square.');
    end
    x = 0:1/(m-1):1;
    [Y,X] = meshgrid(x,x);
    object_texture(:,1) = Y(:);
    object_texture(:,2) = X(end:-1:1);
    fprintf(fid, 'vt %f %f\n', object_texture');
else
    % create dummy vertex texture
    vertext = vertex(:,1:2)*0 - 1;
    % vertex position
    fprintf(fid, '# %d vertex texture\n', size(vertext,1));
    fprintf(fid, 'vt %f %f\n', vertext');
end

% use mtl
fprintf(fid, ['g ' object_name '_export\n']);
mtl_bump_name = 'bump_map';
fprintf(fid, ['usemtl ' mtl_bump_name '\n']);

% face
fprintf(fid, '# %d faces\n', size(face,1));
face_texcorrd = [face(:,1), face(:,1), face(:,2), face(:,2), face(:,3), face(:,3)];
fprintf(fid, 'f %d/%d %d/%d %d/%d\n', face_texcorrd');

fclose(fid);


% MTL generation
if isfield(options, 'nm_file')
    mtl_file = [object_name '.mtl'];
    fid = fopen(mtl_file,'wt');
    if( fid==-1 )
        error('Can''t open the file.');
        return;
    end
    
    Ka = [0.59 0.59 0.59];
    Kd = [0.5 0.43 0.3];
    Ks = [0.6 0.6 0.6];
    d = 1;
    Ns = 2;
    illum = 2;
    
    fprintf(fid, '# write_obj (c) 2004 Gabriel Peyr¨¦\n');
    
    fprintf(fid, 'newmtl %s\n', mtl_bump_name);
    fprintf(fid, 'Ka  %f %f %f\n', Ka);
    fprintf(fid, 'Kd  %f %f %f\n', Kd);
    fprintf(fid, 'Ks  %f %f %f\n', Ks);
    fprintf(fid, 'd  %d\n', d);
    fprintf(fid, 'Ns  %d\n', Ns);
    fprintf(fid, 'illum %d\n', illum);
    fprintf(fid, 'bump %s\n', options.nm_file);
    
    fprintf(fid, '#\n# EOF\n');

    fclose(fid);
end