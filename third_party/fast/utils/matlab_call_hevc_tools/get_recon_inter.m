function [recon_inter, inter_mask, mv_x_map, mv_y_map] = ...
    get_recon_inter(PU_all, prefilt_Y)

N_frames = length(PU_all);
recon_inter = cell(1, N_frames);
inter_mask = cell(1, N_frames);
mv_x_map = cell(1, N_frames);
mv_y_map = cell(1, N_frames);

for t = 1:N_frames
    N_PU = length(PU_all{t});
    recon_inter{t} = zeros(size(prefilt_Y{1}, 1), size(prefilt_Y{1}, 2));
    inter_mask{t} = zeros(size(prefilt_Y{1}, 1), size(prefilt_Y{1}, 2));
    mv_x_map{t} = zeros(size(prefilt_Y{1}, 1), size(prefilt_Y{1}, 2));
    mv_y_map{t} = zeros(size(prefilt_Y{1}, 1), size(prefilt_Y{1}, 2));
    for pu_idx = 1:N_PU
        pu_struct = PU_all{t}(pu_idx);
        if isempty(pu_struct.intra)
            continue;
        end
        if pu_struct.intra == 1
            continue;
        end
        r = pu_struct.y + 1;
        c = pu_struct.x + 1;
        w = pu_struct.w;
        h = pu_struct.h;
        mv_x = pu_struct.mv_x;
        mv_y = pu_struct.mv_y;
        ref_idx = pu_struct.t_r + 1; % idx of the reference frame
        assert(pu_struct.t == t - 1, 'Frame indexing for MC is wrong');
        
        r0 = r + floor(mv_y / 4 + 0.01);
        c0 = c + floor(mv_x / 4 + 0.01);
        refer_patch = get_referenced_patch(prefilt_Y{ref_idx}, ...
            r0, c0, w, h);
        interp_patch = fractional_interpolate(...
            refer_patch, mod(mv_x, 4), mod(mv_y, 4));
        try
            recon_inter{t}(r:(r + h - 1), c:(c + w - 1)) = interp_patch;
        catch
            debug_var = 1;
        end
        inter_mask{t}(r:(r + h - 1), c:(c + w - 1)) = pu_idx;
        
        mv_x_map{t}(r:(r + h - 1), c:(c + w - 1)) = mv_x;
        mv_y_map{t}(r:(r + h - 1), c:(c + w - 1)) = mv_y;
    end
end

end

function ref_patch = get_referenced_patch(input_image, r, c, w, h)
r0 = r - 3;
r1 = r + h - 1 + 4;
c0 = c - 3;
c1 = c + w - 1 + 4;

X = c0:c1;
Y = r0:r1;
X_clip = min(max(X, 1), size(input_image, 2));
Y_clip = min(max(Y, 1), size(input_image, 1));

ref_patch = input_image(Y_clip, X_clip);

% r0_clip = max(r0, 1);
% c0_clip = max(c0, 1);
% r1_clip = min(r1, size(input_image, 1));
% c1_clip = min(c1, size(input_image, 2));

% clip_patch = input_image(r0_clip:r1_clip, c0_clip:c1_clip);
% if r0 < r0_clip
%     clip_patch = padarray(clip_patch, [r0_clip - r0, 0], 'pre', 'replicate');
% end
% 
% if r1 > r1_clip
%     clip_patch = padarray(clip_patch, [r1 - r1_clip, 0], 'post', 'replicate');
% end
% 
% if c0 < c0_clip
%     clip_patch = padarray(clip_patch, [0, c0_clip - c0], 'pre', 'replicate');
% end
% 
% if c1 > c1_clip
%     clip_patch = padarray(clip_patch, [0, c1 - c1_clip], 'post', 'replicate');
% end
% 
% ref_patch = clip_patch;
end

function interp_patch = fractional_interpolate(refer_patch, ...
    mv_x_ind, mv_y_ind)

filter_coeff = [0, 0, 0, 64, 0, 0, 0, 0; ...
    -1, 4, -10, 58, 17, -5, 1, 0; ...
    -1, 4, -11, 40, 40, -11, 4, -1; ...
    0, 1, -5, 17, 58, -10, 4, -1];

patch_1 = conv2(refer_patch, ...
    rot90(filter_coeff(mv_x_ind + 1, :), 2), 'valid');
patch_2 = conv2(patch_1, ...
    rot90(filter_coeff(mv_y_ind + 1, :), 2)', 'valid');

interp_patch = floor(patch_2 / 64 + 0.01);

interp_patch = round(interp_patch / 64);

interp_patch = max(min(interp_patch, 255), 0);

end