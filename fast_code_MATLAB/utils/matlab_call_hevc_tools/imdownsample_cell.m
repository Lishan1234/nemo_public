function rgb_half_cell = imdownsample_cell(rgb_cell, ratio, clip_dim)

rgb_half_cell = cell(size(rgb_cell));
for i = 1:length(rgb_cell)
    rgb_half_cell{i} = imresize(rgb_cell{i}, 1 / ratio);
    if exist('clip_dim', 'var')
        height_clip = floor(size(rgb_half_cell{i}, 1) / clip_dim) ...
            * clip_dim;
        width_clip = floor(size(rgb_half_cell{i}, 2) / clip_dim) ...
            * clip_dim;
        rgb_half_cell{i} = rgb_half_cell{i}(1:height_clip, ...
            1:width_clip, :);
    end
end

end