clc;
clear;
close all;

result_path = fullfile(cd, '..', 'results');
if ~exist(result_path, 'dir')
    mkdir(result_path);
end

temp_path = fullfile(cd, '..', 'temp_data');
if ~exist(temp_path, 'dir')
    mkdir(temp_path);
end

% ------------------------------------------------------------------------
% Add the paths of the sub-directories to the system path, and compile the
% mex if needed
% ------------------------------------------------------------------------
set_path;

recompile_mex = 1;
if recompile_mex == 1
    compile_mex_hevc_sr;
end

% ------------------------------------------------------------------------
% Configurations
% ------------------------------------------------------------------------
seq_name = 'starcraft';
num_frames = 48; % Number of frames to play with
gop = 8;
clip_dim = 8;
QP = 27; % Quantization level!

% Although this is left as a parameter, there are few components where only
% sr_ratio = 2 is supported, right now.
sr_ratio = 3; 

b_encode = 1; % If we've encoded once, we don't need to encode the video again.
b_recall_sr = 1; % If we've called SR on this sequence once, we don't need to call it again.
b_make_plot = 1; % Do we need plots or just numbers?

% ------------------------------------------------------------------------
% Load images from folder
% ------------------------------------------------------------------------
enc_params = make_encoding_param('num_frames', num_frames, 'QP', QP);
folder_path = fullfile(enc_params.test_yuv_dir, seq_name);

sr_high_rgb = cell(1, num_frames);
bicubic_high_rgb = cell(1, num_frames);
original_high_rgb = cell(1, num_frames);
original_low_rgb = cell(1, num_frames);

 for i=1:num_frames
    if i < 10
       file_name = strcat('000', int2str(i), '.png');
    elseif i < 100
       file_name = strcat('00', int2str(i), '.png');
    else
       file_name = strcat('0', int2str(i), '.png');
    end
    disp(file_name);
    sr_high_rgb{i} = imread(fullfile(folder_path,'sr', file_name));
    bicubic_high_rgb{i} = imread(fullfile(folder_path,'bicubic', file_name));
    original_high_rgb{i} = imread(fullfile(folder_path,'original_high', file_name));
    original_low_rgb{i} = imread(fullfile(folder_path,'original_low', file_name));
 end

sr_high_y = rgb2y_cell(sr_high_rgb);
bicubic_high_y = rgb2y_cell(bicubic_high_rgb);
original_high_y = rgb2y_cell(original_high_rgb);
original_low_y = rgb2y_cell(original_low_rgb);
    

% ------------------------------------------------------------------------
% FAST algorithms
% ------------------------------------------------------------------------
number_of_cycles = num_frames / gop;
fileID = fopen(string(seq_name)+ "_" +int2str(num_frames) +"_log.txt",'w');
PSNR_transfer = zeros(1, num_frames);
PSNR_bicubic = zeros(1, num_frames);
PSNR_SR = zeros(1, num_frames);

%runtime_transfer = zeros(1, num_frames);
%runtime_deblock = zeros(1, num_frames);
runtime_1 = zeros(1, num_frames);
runtime_2 = zeros(1, num_frames);
runtime_3 = zeros(1, num_frames);
runtime_4 = zeros(1, num_frames);
runtime_5 = zeros(1, num_frames);
runtime_6 = zeros(1, num_frames);

PU_block_number = zeros(1, num_frames);
percent = zeros(1, num_frames);

original_low_rgb_cycle = cell(1, gop);

for i= 1:number_of_cycles
    sr_index = (i-1) * gop + 1;
    sr_image = sr_high_y{sr_index};
    
    for j=1:gop
        original_low_rgb_cycle{j} = original_low_rgb{sr_index + j-1};
    end
    
    enc_info = encode_sequence_from_cell(original_low_rgb_cycle, seq_name, enc_params);
    dec_info = get_dumped_information(enc_params, enc_info);
    [intra_recon, inter_mc, res_all, inter_mask, other_info] = ...
        parse_all_saved_info(dec_info);
 
    hevc_info = struct('intra_recon', {intra_recon}, ...
        'inter_mc', {inter_mc}, ...
        'res_all', {res_all}, ...
        'inter_mask', {inter_mask}, ...
        'other_info', {other_info});
    
    [imgs_h_transfer, anyting_else, time_info, percent_transfer] = hevc_transfer_sr(...
    sr_image, gop, hevc_info, sr_ratio);
    
    for j = 1:gop
        PSNR_transfer(sr_index + j-1) = ...
            computePSNR(original_high_y{sr_index + j-1}, imgs_h_transfer{j});
        %runtime_transfer(sr_index + j-1) = time_info.runtime_transfer(j);
        %runtime_deblock(sr_index + j-1) = time_info.runtime_deblock(j);
        runtime_1(sr_index + j-1) = time_info.runtime_1(j);
        runtime_2(sr_index + j-1) = time_info.runtime_2(j);
        runtime_3(sr_index + j-1) = time_info.runtime_3(j);
        runtime_4(sr_index + j-1) = time_info.runtime_4(j);
        runtime_5(sr_index + j-1) = time_info.runtime_5(j);
        runtime_6(sr_index + j-1) = time_info.runtime_6(j);
        
        PU_block_number(sr_index + j-1) = anyting_else.block_number(j);
        percent(sr_index + j-1) = percent_transfer(j);
    end
    
    for j=1:gop
        index = sr_index + j-1;
        if index < 10
            file_name = strcat('000', int2str(index), '.bmp');
        elseif index < 100
            file_name = strcat('00', int2str(index), '.bmp');
        else
            file_name = strcat('0', int2str(index), '.bmp');
        end
        imwrite(uint8(imgs_h_transfer{j}), sprintf(fullfile(folder_path,'result', file_name)), 'BMP');
    end
end

% ------------------------------------------------------------------------
% Compute PSNR
% ------------------------------------------------------------------------

for i=1:num_frames
    PSNR_bicubic(i) = computePSNR(original_high_y{i}, bicubic_high_y{i});
    PSNR_SR(i) = computePSNR(original_high_y{i}, sr_high_y{i});
end

figure
axis([0 120 18 30])
plot(PSNR_bicubic, 'g');
hold on
plot(PSNR_transfer, 'b--o');
hold on
plot(PSNR_SR, 'r');

% ------------------------------------------------------------------------
% Save as log file
% ------------------------------------------------------------------------
fileID = fopen(string(seq_name)+ "_" +int2str(gop)...
    + "_" +"x"+int2str(sr_ratio)+"_log.txt",'w');

%{
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "transfer runtime", runtime_transfer);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "deblock runtime", runtime_deblock);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "PSNR_transfer", PSNR_transfer);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "PSNR_SR", PSNR_SR);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "PSNR_bicubic", PSNR_bicubic);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "percent transfer", percent);
fprintf(fileID, '%s %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d\r\n',...
    "PU block number", PU_block_number);
%}

fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "runtime_1", runtime_1);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "runtime_2", runtime_2);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "runtime_3", runtime_3);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "runtime_4", runtime_4);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "runtime_5", runtime_5);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "runtime_6", runtime_6);
fclose(fileID);