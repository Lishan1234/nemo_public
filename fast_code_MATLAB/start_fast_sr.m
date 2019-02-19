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
group_of_pictures = 8;
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
% Bicubic PSNR
% ------------------------------------------------------------------------
imgs_h_bicubic = cell(1, num_frames);
PSNR_bicubic = zeros(1, num_frames);
for i = 1:num_frames
    imgs_h_bicubic{i} = imresize(Y_low_res{i}, [1080, 1920], 'bicubic');
    PSNR_bicubic(i) = computePSNR(Y_high_res_gt{i}, imgs_h_bicubic{i});
end

% ------------------------------------------------------------------------
% Super Resolution PSNR
% ------------------------------------------------------------------------
PSNR_SR = zeros(1, num_frames);
for i = 1:num_frames
    PSNR_SR(i) = computePSNR(Y_high_res_gt{i}, Y_high_res_gt{i});
end


% ------------------------------------------------------------------------
% FAST algorithms
% ------------------------------------------------------------------------
number_of_cycles = num_frames / group_of_pictures;
fileID = fopen(string(seq_name)+ "_" +int2str(num_frames) +"_log.txt",'w');
PSNR_transfer = zeros(1, num_frames);
runtime_transfer = zeros(1, num_frames);
runtime_deblock = zeros(1, num_frames);
PU_block_number = zeros(1, num_frames);
percent = zeros(1, num_frames);

for i= 1:number_of_cycles
    sr_index = (i-1) * group_of_pictures + 1;
    if sr_index < 10
        sr_file_name = strcat('000', int2str(sr_index), '.png');
    elseif sr_index < 100
        sr_file_name = strcat('00', int2str(sr_index), '.png');
    else
        sr_file_name = strcat('0', int2str(sr_index), '.png');
    end
    image_path = fullfile(folder_path,'sr', sr_file_name);
    disp(image_path);
    
    sr_image = imread(image_path);
    rgb_vec = reshape(sr_image, img_height * img_width, 3);
    yuv_vec = convertRgbToYuv(rgb_vec);
    sr_image = reshape(yuv_vec(:, 1), img_height, img_width);
    
    PU_cycle = cell(1, group_of_pictures);
    TU_cycle = cell(1, group_of_pictures);
    mv_x_cycle = cell(1, group_of_pictures);
    mv_y_cycle = cell(1, group_of_pictures);
    
    intra_recon_cycle = cell(1, group_of_pictures);
    inter_mc_cycle = cell(1, group_of_pictures);
    res_all_cycle = cell(1, group_of_pictures);
    inter_mask_cycle = cell(1, group_of_pictures);
    
    for j=1:group_of_pictures
        PU_cycle{j} = other_info.PU{sr_index + j-1};
        TU_cycle{j} = other_info.TU{sr_index + j-1};
        mv_x_cycle{j} = other_info.mv_x{sr_index + j-1};
        mv_y_cycle{j} = other_info.mv_y{sr_index + j-1};
        
        intra_recon_cycle{j} = intra_recon{sr_index + j-1};
        inter_mc_cycle{j} = inter_mc{sr_index + j-1};
        res_all_cycle{j} = res_all{sr_index + j-1};
        inter_mask_cycle{j} = inter_mask{sr_index + j-1};
    end
    
    other_info_cycle = struct('PU', {PU_cycle}, ...
        'TU', {TU_cycle}, ...
        'mv_x', {mv_x_cycle}, ...
        'mv_y', {mv_x_cycle});
    
    hevc_info_cycle = struct('intra_recon', {intra_recon_cycle}, ...
        'inter_mc', {inter_mc_cycle}, ...
        'res_all', {res_all_cycle}, ...
        'inter_mask', {inter_mask_cycle}, ...
        'other_info', {other_info_cycle});
    
    [imgs_h_transfer, anyting_else, time_info, percent_transfer] = hevc_transfer_sr(...
    sr_image, group_of_pictures, hevc_info_cycle);
    
    for j = 1:group_of_pictures
        PSNR_transfer((i-1)*group_of_pictures+j) = ...
            computePSNR(Y_high_res_gt{(i-1)*group_of_pictures+j}, imgs_h_transfer{j});
        runtime_transfer((i-1)*group_of_pictures+j) = time_info.runtime_transfer(j);
        runtime_deblock((i-1)*group_of_pictures+j) = time_info.runtime_deblock(j);
        PU_block_number((i-1)*group_of_pictures+j) = anyting_else.block_number(j);
        percent((i-1)*group_of_pictures+j) = percent_transfer(j);
    end
end

figure
axis([0 120 18 30])
plot(PSNR_bicubic);
hold on
plot(PSNR_transfer);
hold on
plot(PSNR_SR);


% ------------------------------------------------------------------------
% Save as log file
% ------------------------------------------------------------------------
fileID = fopen(string(seq_name)+ "_" +int2str(num_frames) +"_log.txt",'w');

fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "transfer runtime", runtime_transfer);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "deblock runtime", runtime_deblock);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "PSNR_transfer", PSNR_transfer);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "PSNR_bicubic", PSNR_bicubic);
fprintf(fileID, '%s %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\r\n',...
    "percent transfer", percent);
fprintf(fileID, '%s %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d %6d\r\n',...
    "PU block number", PU_block_number);

fclose(fileID);