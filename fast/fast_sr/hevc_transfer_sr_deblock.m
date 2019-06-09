function [img_h_transfer, other_info] = hevc_transfer_sr_deblock(...
    sr_results, hevc_info, params)

if ~exist('params', 'var')
    params = [];
end

if ~isfield(params, 'transfer_thresh')
    params.transfer_thresh = 10;
end

if ~isfield(params, 'deblock')
    params.deblock = 1;
end

if ~isfield(params, 'QP')
    params.QP = 47;
end

b_debug = 0;
% ------------------------------------------------------------------------
% Parse Parameters
% ------------------------------------------------------------------------
N_frames = length(sr_results);
img_width = size(sr_results{1}, 2);
img_height = size(sr_results{1}, 1);

% ------------------------------------------------------------------------
% Reconstruct low-resolution image
% ------------------------------------------------------------------------
recon_together = combine_dumped_info(hevc_info.intra_recon, ...
    hevc_info.inter_mc, hevc_info.res_all, hevc_info.inter_mask);

% ------------------------------------------------------------------------
% Frame-by-frame transfer of high-resolution image
% ------------------------------------------------------------------------
img_h_transfer = cell(1, N_frames);
other_info = [];
if params.deblock == 1
    other_info.img_h_transfer_nodeblock = cell(1, N_frames);
end

% I-frame
img_h_transfer{1} = sr_results{1};
other_info.img_h_transfer_nodeblock{1} = sr_results{1};

% Perform a single threshold on the residue!

% P-frames
for f_idx = 2:N_frames
    tic;
    img_h_transfer{f_idx} = zeros(img_height, img_width);
    PU_now = hevc_info.other_info.PU{f_idx};
    TU_now = hevc_info.other_info.TU{f_idx};
    
    % Upsample the residue blockwise with TU block structure!
    res_h = Blockwise_upsample_with_TU(hevc_info.res_all{f_idx}, TU_now);
    
    for pu_idx = 1:length(hevc_info.other_info.PU{f_idx})
        if isempty(PU_now(pu_idx).x) || isempty(PU_now(pu_idx).w)
            continue;
        end
        x_l = PU_now(pu_idx).x;
        y_l = PU_now(pu_idx).y;
        x_h = 2 * x_l;
        y_h = 2 * y_l;
        w = PU_now(pu_idx).w;
        h = PU_now(pu_idx).h;
        
        switch PU_now(pu_idx).intra
            case 0
                % Inter prediction!
                
                % Decide whether to transfer or not by residue!
                res_l_patch = hevc_info.res_all{f_idx}...
                    ((y_l + 1):(y_l + h), (x_l + 1):(x_l + w));
                mean_ssd = sum(abs(res_l_patch(:))) / numel(res_l_patch);
                if mean_ssd < params.transfer_thresh
                    % Perform transfer
                    mv_x_h = 2 * PU_now(pu_idx).mv_x / 4;
                    mv_y_h = 2 * PU_now(pu_idx).mv_y / 4;
                    f_ref = PU_now(pu_idx).t_r + 1;
                    
                    ref_h_patch = sr_interpolate(img_h_transfer{f_ref}, ...
                        x_h, y_h, 2 * w, 2 * h, mv_x_h, mv_y_h);
%                     ref_h_patch = subpix_interp(img_h_transfer{f_ref}, ...
%                         (x_h + 1 + mv_x_h):(x_h + mv_x_h + 2 * w), ...
%                         (y_h + 1 + mv_y_h):(y_h + mv_y_h + 2 * h));
                    
                    res_h_patch = res_h((y_h + 1):(y_h + 2 * h), ...
                        (x_h + 1):(x_h + 2 * w));
                    
                    
                    img_h_transfer{f_idx}((y_h + 1):(y_h + 2 * h), ...
                        (x_h + 1):(x_h + 2 * w)) = ...
                        ref_h_patch + res_h_patch;
                    if any(isnan(ref_h_patch(:))) || any(isnan(res_h_patch(:)))
                        db_var = 1;
                    end
                else
                    % Otherwise: Prediction error is too large, perform
                    % bicubic interpolation on the low-res reconstruction
                    inter_patch = recon_together{f_idx}...
                        ((y_l + 1):(y_l + h), (x_l + 1):(x_l + w));
                    img_h_transfer{f_idx}((y_h + 1):(y_h + 2 * h), ...
                        (x_h + 1):(x_h + 2 * w)) = imresize(inter_patch, 2, 'bicubic');
                    
                    if any(isnan(inter_patch(:)))
                        db_var = 1;
                    end
                end
            case 1
                % Intra prediction: Copy the intra prediction results, and
                % SR!!
                intra_patch = recon_together{f_idx}((y_l + 1):(y_l + h), ...
                    (x_l + 1):(x_l + w));
                if any(isnan(intra_patch(:)))
                    db_var = 1;
                end
                img_h_transfer{f_idx}((y_h + 1):(y_h + 2 * h), ...
                    (x_h + 1):(x_h + 2 * w)) = imresize(intra_patch, 2, 'bicubic');
            otherwise
                error('Intra indicator can only be 0 or 1 in PU!');
        end
        
    end
    
    t1 = toc;
    tic;
    
    
    % --------------------------------------------------------------------
    % Perform deblock if needed
    % --------------------------------------------------------------------
    if params.deblock == 1
        % Perform deblocking on img_h_transfer{f_idx} according to PU and
        % TU structure.
        
        img_before_deblock = img_h_transfer{f_idx};
        if b_debug == 1
            imwrite(uint8(img_before_deblock), fullfile(cd, '..',  ...
                'temp_data', sprintf('deblock_input_%d.bmp', f_idx)), ...
                'BMP');
        end
        
        % Generate the PU and TU boundary
       
        % Deblock the image vertically
        deblock_v_all = vertical_edge_deblock(...
            img_before_deblock, params.QP);
        
        deblock_v = deblock_v_all;

        % Deblock the image horizontally
        deblock_h_all = vertical_edge_deblock(deblock_v', params.QP)';
        deblock_h = deblock_h_all;
        
        % Update the results
        img_h_transfer{f_idx} = deblock_h;
        other_info.img_h_transfer_nodeblock{f_idx} = img_before_deblock;
    end
    t2 = toc;
    
    t_deblock = t2 / (t1 + t2)
end

end