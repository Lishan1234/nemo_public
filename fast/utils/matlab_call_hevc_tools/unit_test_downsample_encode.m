% This script tests the pipeline to call the encoder, decoder of HEVC, and
% parse the dumped information to MATLAB.

clc;
clear;
close all;

% ------------------------------------------------------------------------
% Basic setup
% ------------------------------------------------------------------------
seq_name = 'BQSquare';
num_frames = 3;
enc_params = make_encoding_param('num_frames', num_frames);
clip_dim = 16;

% ------------------------------------------------------------------------
% Read the sequence and downsample it!
% ------------------------------------------------------------------------
[yuv_filename, img_width, img_height] = ...
    get_file_info_in_video_test_set(enc_params.test_yuv_dir, seq_name);
rgb_cell = load_rgb_cell_from_yuv(fullfile(enc_params.test_yuv_dir, ...
    yuv_filename), img_width, img_height, num_frames);

% ------------------------------------------------------------------------
% Downsample the image
% ------------------------------------------------------------------------
rgb_half_cell = imdownsample_cell(rgb_cell, 2, clip_dim);

% ------------------------------------------------------------------------
% Call the pipeline
% ------------------------------------------------------------------------
enc_info = encode_sequence_from_cell(rgb_half_cell, seq_name, enc_params);
dec_info = get_dumped_information(enc_params, enc_info);
[intra_recon, inter_mc, res_all, inter_mask, other_info] = ...
    parse_all_saved_info(dec_info);
recon_together = combine_dumped_info(intra_recon, inter_mc, res_all, ...
    inter_mask);

% ------------------------------------------------------------------------
% Load the actual decoded video
% ------------------------------------------------------------------------
ref_Y = load_Y_of_yuv(dec_info.enc_info.yuv_recon_name, ...
    dec_info.enc_info.img_width, dec_info.enc_info.img_height, num_frames);

% ------------------------------------------------------------------------
% Verify the results: Blockwise Comparison with CodecVisa
% ------------------------------------------------------------------------
% Variables to compare: Intra Recon vs Final, Residue vs True Residue,
% Inter prediction vs MC, motion vector
% Frames to compare: 1, 2, 3
%
% Unfortunately, this has to be done interactively...

frame_idx = 3;
cu_x = 160;
cu_y = 40;
w = 8;
h = 8;

% is_intra = '?';
intra_final = intra_recon{frame_idx};
mc_pred = inter_mc{frame_idx};
res = res_all{frame_idx};
recon_frame = recon_together{frame_idx};
recon_gt = ref_Y{frame_idx};
mv_x = other_info.mv_x{frame_idx};
mv_y = other_info.mv_y{frame_idx};

fprintf('-------------------------------------------------------------\n');
fprintf('intra final \n');
blk_intra = intra_final((cu_y + 1):(cu_y + h), (cu_x + 1):(cu_x + w))

fprintf('-------------------------------------------------------------\n');
fprintf('inter mc\n');
blk_inter_mc = mc_pred((cu_y + 1):(cu_y + h), (cu_x + 1):(cu_x + w))

fprintf('-------------------------------------------------------------\n');
fprintf('Residue \n');
blk_res = res((cu_y + 1):(cu_y + h), (cu_x + 1):(cu_x + w))

fprintf('-------------------------------------------------------------\n');
fprintf('Recon all\n');
blk_final = recon_frame((cu_y + 1):(cu_y + h), (cu_x + 1):(cu_x + w))

fprintf('-------------------------------------------------------------\n');
fprintf('Motion vector\n');
mx = mv_x((cu_y + 1):(cu_y + h), (cu_x + 1):(cu_x + w))
my = mv_y((cu_y + 1):(cu_y + h), (cu_x + 1):(cu_x + w))