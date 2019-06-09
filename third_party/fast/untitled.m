clc;
clear;
close all;

num_frames = 120;
original_1080p_rgb = cell(1, num_frames);
folder_path = fullfile('/Users/sunghwan/Desktop/fast_code_MATLAB/data', 'starcraft');

 for i=1:num_frames
    if i < 10
       file_name = strcat('000', int2str(i), '.png');
    elseif i < 100
       file_name = strcat('00', int2str(i), '.png');
    else
       file_name = strcat('0', int2str(i), '.png');
    end
    original_1080p_rgb{i} = imread(fullfile(folder_path,'original_high', file_name));
    
 end
 
original_high_y = rgb2y_cell(original_1080p_rgb);

for i=1:num_frames
    if i < 10
       file_name = strcat('000', int2str(i), '.bmp');
    elseif i < 100
       file_name = strcat('00', int2str(i), '.bmp');
    else
       file_name = strcat('0', int2str(i), '.bmp');
    end
    imwrite(original_high_y{i}, fullfile(folder_path,'original', 'black', file_name));
end