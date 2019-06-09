function final_Y = load_Y_of_yuv(filename, img_width, img_height, N_frames)

final_Y = cell(1, N_frames);
for f_idx = 1:N_frames
    [~, ~, yuv_img] = loadFileYuv(filename, img_width, img_height, f_idx);
    final_Y{f_idx} = yuv_img(:, :, 1);
end

end