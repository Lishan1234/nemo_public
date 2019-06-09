clc;
clear;
close all;

num_frames = 48;
sr_1080p_rgb = cell(1, num_frames);
bicubic_1080p_rgb = cell(1, num_frames);
original_1080p_rgb = cell(1, num_frames);
original_540p_rgb = cell(1, num_frames);
folder_path = fullfile('/Users/sunghwan/Desktop/fast_code_MATLAB/data', 'starcraft');

 for i=1:num_frames
    if i < 10
       file_name = strcat('000', int2str(i), '.png');
    elseif i < 100
       file_name = strcat('00', int2str(i), '.png');
    else
       file_name = strcat('0', int2str(i), '.png');
    end
    sr_1080p_rgb{i} = imread(fullfile(folder_path,'sr', file_name));
    bicubic_1080p_rgb{i} = imread(fullfile(folder_path,'bicubic', file_name));
    original_1080p_rgb{i} = imread(fullfile(folder_path,'original_high', file_name));
    original_540p_rgb{i} = imread(fullfile(folder_path,'original_low', file_name));

    sr_1080p_rgb{i} = imcrop(sr_1080p_rgb{i},[0 0 1920 1056]);
    bicubic_1080p_rgb{i} = imcrop(bicubic_1080p_rgb{i},[0 0 1920 1056]);
    original_1080p_rgb{i} = imcrop(original_1080p_rgb{i},[0 0 1920 1056]);
    original_540p_rgb{i} = imcrop(original_540p_rgb{i},[0 0 480 264]);
    
    imwrite(sr_1080p_rgb{i}, fullfile(folder_path,'sr', file_name));
    imwrite(bicubic_1080p_rgb{i}, fullfile(folder_path,'bicubic', file_name));
    imwrite(original_1080p_rgb{i}, fullfile(folder_path,'original_high', file_name));
    imwrite(original_540p_rgb{i}, fullfile(folder_path,'original_low', file_name));
 end
