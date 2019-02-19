function rgb_cell = load_rgb_cell_from_yuv(yuv_path, img_width, ...
    img_height, num_frames)

rgb_cell = cell(1, num_frames);

for i = 1:num_frames
    [~, rgb_cell{i}, ~] = loadFileYuv(yuv_path, img_width, img_height, i);
end

end