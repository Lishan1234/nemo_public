function patch = sr_interpolate(img_h_sr, x, y, w, h, mv_x, mv_y)
% w, h are real fractional!
xp = x + floor(mv_x + 0.01);
yp = y + floor(mv_y + 0.01);
ref_patch = get_reference_patch_sr(img_h_sr, yp + 1, xp + 1, w, h);

mv_x_ind = mod(round(mv_x * 4), 4);
mv_y_ind = mod(round(mv_y * 4), 4);
patch = fractional_interpolate_sr(ref_patch, ...
    mv_x_ind, mv_y_ind);

% [X, Y] = meshgrid(3 + mv_x_ind / 4 + (1:w), 3 + mv_y_ind / 4 + (1:h));
% patch = min(max(round(interp2(ref_patch, X, Y, 'cubic')), 0), 255);

end

function ref_patch = get_reference_patch_sr(input_image, r, c, w, h)
r0 = r - 3;
r1 = r + h - 1 + 4;
c0 = c - 3;
c1 = c + w - 1 + 4;

X = c0:c1;
Y = r0:r1;
X_clip = min(max(X, 1), size(input_image, 2));
Y_clip = min(max(Y, 1), size(input_image, 1));

ref_patch = input_image(Y_clip, X_clip);

% 
% r0_clip = max(r0, 1);
% c0_clip = max(c0, 1);
% r1_clip = min(r1, size(input_image, 1));
% c1_clip = min(c1, size(input_image, 2));
% 
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

function interp_patch = fractional_interpolate_sr(refer_patch, ...
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