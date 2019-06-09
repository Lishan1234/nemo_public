function [seq_filename, img_width, img_height] = ...
    get_file_info_in_video_test_set(test_dir, seq_name)

dir_ent = dir(fullfile(test_dir, [seq_name, '*.yuv']));
if isempty(dir_ent)
    error('Could not find the file for the sequence!');
end
seq_filename = dir_ent(1).name;
[~, img_width, img_height] = parse_test_yuv_name(seq_filename);

end