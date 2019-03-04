function Y_cell = rgb2y_cell(rgb_cell)
Y_cell = cell(size(rgb_cell));

img_height = size(rgb_cell{1}, 1);
img_width = size(rgb_cell{1}, 2);

for i = 1:numel(rgb_cell)
    rgb_vec = reshape(rgb_cell{i}, img_height * img_width, 3);
    yuv_vec = convertRgbToYuv(rgb_vec);
    Y_cell{i} = reshape(yuv_vec(:, 1), img_height, img_width);
end

end