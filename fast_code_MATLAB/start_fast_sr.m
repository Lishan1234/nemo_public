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
seq_name = 'soccer';
num_frames = 4; % Number of frames to play with
clip_dim = 16;
QP = 27; % Quantization level!

% Although this is left as a parameter, there are few components where only
% sr_ratio = 2 is supported, right now.
sr_ratio = 2; 

method_name = 'CNN'; % Use SRCNN as benchmark method

b_encode = 1; % If we've encoded once, we don't need to encode the video again.
b_recall_sr = 1; % If we've called SR on this sequence once, we don't need to call it again.
b_make_plot = 1; % Do we need plots or just numbers?

% ------------------------------------------------------------------------
% Call HEVC to encode the sequence, and get the syntax elements
% ------------------------------------------------------------------------
save_file = fullfile(cd, '..', 'temp_data', ...
    sprintf('%s_info.mat', seq_name)); % 

if b_encode == 1 || ~exist(save_file, 'file')
    % This function specifies where the HEVC dataset folder is
    enc_params = make_encoding_param('num_frames', num_frames, 'QP', QP);
    folder_path = fullfile(enc_params.test_yuv_dir, seq_name);
    [yuv_filename, img_width, img_height] = ...
        get_file_info_in_video_test_set(folder_path, seq_name);

    % Load the RGB cells
    rgb_cell = load_rgb_cell_from_yuv(fullfile(folder_path ,...
        yuv_filename), img_width, img_height, num_frames);
    Y_high_res_gt = rgb2y_cell(rgb_cell);
    
    % Downsample the video sequence by 2
    rgb_half_cell = imdownsample_cell(rgb_cell, 2, clip_dim);
    
    % Call the encoding function
    enc_info = encode_sequence_from_cell(rgb_half_cell, seq_name, enc_params);
    
    % Get the syntax elements
    dec_info = get_dumped_information(enc_params, enc_info);
    [intra_recon, inter_mc, res_all, inter_mask, other_info] = ...
        parse_all_saved_info(dec_info);
    
    % Get the compressed low-resolution video
    Y_low_res = load_Y_of_yuv(dec_info.enc_info.yuv_recon_name, ...
        dec_info.enc_info.img_width, dec_info.enc_info.img_height, num_frames);
    
    % Data structure to hold all of the compressed information
    hevc_info = struct('intra_recon', {intra_recon}, ...
        'inter_mc', {inter_mc}, ...
        'res_all', {res_all}, ...
        'inter_mask', {inter_mask}, ...
        'other_info', {other_info});
    
    % Save them, so that we do not encode the same sequence again, in case
    % there is a bug in this script later.
    save(save_file, 'Y_low_res', 'rgb_cell', 'Y_high_res_gt', ...
        'dec_info', 'hevc_info');
else
    load(save_file, 'Y_low_res', 'rgb_cell', 'Y_high_res_gt', ...
        'dec_info', 'hevc_info');
end

% ------------------------------------------------------------------------
% Get Super Resolution Image
% ------------------------------------------------------------------------
image_path = fullfile(folder_path,'sr','0001.png');
sr_image = imread(image_path);
sr_image = double(sr_image);



% ------------------------------------------------------------------------
% FAST algorithms
% ------------------------------------------------------------------------
[imgs_h_transfer, other_info, time_info, percent_transfer] = hevc_transfer_sr(...
    sr_image, num_frames, hevc_info);


% ------------------------------------------------------------------------
% Save as log file
% ------------------------------------------------------------------------
fileID = fopen(string(seq_name)+ "_" +int2str(num_frames) +"_log.txt",'w');

fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "transfer runtime", time_info.runtime_transfer);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "deblock runtime", time_info.runtime_deblock);
fprintf(fileID, '%s %6d %6d %6d %6d %6d %6d %6d %6d\r\n',...
    "PU block number", other_info.block_number);
fprintf(fileID, '%s %6d %6d %6d %6d %6d %6d %6d %6d\r\n',...
    "percent transfer", percent_transfer);
fclose(fileID);