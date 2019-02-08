function [intra_recon, inter_mc, recon_res, inter_mask, other_info] = ...
    parse_all_saved_info(dec_info)

% ------------------------------------------------------------------------
% Parse the dumped.txt
% ------------------------------------------------------------------------
[PU_all, res_luma_all] = parse_dumped_coeff_multiple_poc(...
    dec_info.dump_txt_name);
img_width = dec_info.enc_info.img_width;
img_height = dec_info.enc_info.img_height;
N_frames = length(PU_all);

% ------------------------------------------------------------------------
% For DEBUG ONLY!
% ------------------------------------------------------------------------
db_var = 0;
if db_var == 1
    if ~exist(fullfile(cd, 'temp_data'), 'dir')
        mkdir(fullfile(cd, 'temp_data'));
    end
    
    save(fullfile(cd, 'temp_data', 'db_parse_all_saved_info.mat'), ...
        'dec_info', 'PU_all', 'res_luma_all', 'img_width', 'img_height', ...
        'N_frames');
    load(fullfile(cd, 'temp_data', 'db_parse_all_saved_info.mat'), ...
        'dec_info', 'PU_all', 'res_luma_all', 'img_width', 'img_height', ...
        'N_frames');
end

% ------------------------------------------------------------------------
% Load the decoded sequence
% ------------------------------------------------------------------------
final_Y = load_Y_of_yuv(dec_info.enc_info.yuv_recon_name, img_width, ...
    img_height, N_frames);

% ------------------------------------------------------------------------
% Reconstruct Residue
% ------------------------------------------------------------------------
recon_res = cell(1, N_frames);
for f_idx = 1:N_frames
    recon_res{f_idx} = reconstruct_res(res_luma_all{f_idx}, ...
        [img_height, img_width]);
end

% ------------------------------------------------------------------------
% Reconstruct inter
% ------------------------------------------------------------------------
[inter_mc, inter_mask, mv_x_map, mv_y_map] = ...
    get_recon_inter(PU_all, final_Y);

% ------------------------------------------------------------------------
% Reconstruct intra
% ------------------------------------------------------------------------
intra_recon = cell(1, N_frames);
for f_idx = 1:N_frames
    intra_recon{f_idx} = zeros(img_height, img_width);
    intra_mask_frame = inter_mask{f_idx} == 0;
    intra_recon{f_idx}(intra_mask_frame) = final_Y{f_idx}(intra_mask_frame);
end

% ------------------------------------------------------------------------
% Output Other Information
% ------------------------------------------------------------------------
other_info = struct('PU', {PU_all}, 'TU', {res_luma_all}, ...
    'mv_x', {mv_x_map}, 'mv_y', {mv_y_map});

end