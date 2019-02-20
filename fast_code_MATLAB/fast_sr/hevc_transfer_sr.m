function [img_h_transfer, other_info, time_info, percent_transfer] = hevc_transfer_sr(...
    sr_result, N_frames, hevc_info, params)

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
img_width = size(sr_result, 2);
img_height = size(sr_result, 1);

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
time_info = [];

num_transfer = 0;
num_bicubic = 0; 
percent_transfer = zeros(1, N_frames);

% I-frame
img_h_transfer{1} = sr_result;
other_info.block_number = zeros(1, N_frames);

% ------------------------------------------------------------------------
% measure running time
% ------------------------------------------------------------------------
time_info.runtime_transfer = zeros(1, N_frames);
time_info.runtime_deblock = zeros(1, N_frames);


% P-frames

for f_idx = 2:N_frames
    tic;
    img_h_transfer{f_idx} = zeros(img_height, img_width);
    PU_now = hevc_info.other_info.PU{f_idx};
    TU_now = hevc_info.other_info.TU{f_idx};
    
    % Upsample the residue blockwise with TU block structure!
    res_h = Blockwise_upsample_with_TU(hevc_info.res_all{f_idx}, TU_now);
    
    for pu_idx = 1:length(hevc_info.other_info.PU{f_idx})
        other_info.block_number(f_idx) = other_info.block_number(f_idx) + 1;
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
                num_transfer = num_transfer + 1;
                % Decide whether to transfer or not by residue!
                res_l_patch = hevc_info.res_all{f_idx}...
                    ((y_l + 1):(y_l + h), (x_l + 1):(x_l + w));
                %                 dx_patch = conv2(res_l_patch, [-1, 1], 'valid');
                %                 dy_patch = conv2(res_l_patch, [-1; 1], 'valid');
                %                 mean_diff = (sum(abs(dx_patch(:))) + ...
                %                     sum(abs(dy_patch(:)))) / numel(res_l_patch);
                mean_ssd = sum(abs(res_l_patch(:))) / numel(res_l_patch);
                if mean_ssd < params.transfer_thresh
                    %                 if mean_diff < params.transfer_thresh
                    % Perform transfer
                    mv_x_h = 2 * PU_now(pu_idx).mv_x / 4;
                    mv_y_h = 2 * PU_now(pu_idx).mv_y / 4;
                    %temp = PU_now(pu_idx).t_r + 1;
                    %while temp > N_frames
                    %    temp = temp - N_frames;
                    %end
                    %f_ref 가져오는 알고리즘 다시 짜야함
                    
                    f_ref = f_idx-1;
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
                num_bicubic = num_bicubic + 1;
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
    time_info.runtime_transfer(f_idx) = toc;
    percent_transfer(f_idx) = (num_transfer / (num_transfer + num_bicubic)) * 100;
    % --------------------------------------------------------------------
    % Perform deblock if needed
    % --------------------------------------------------------------------
    tic;
    if params.deblock == 1
        % Perform deblocking on img_h_transfer{f_idx} according to PU and
        % TU structure.
        
        % Deblock the image vertically
        deblock_v_all = deblock_mex(int32(img_h_transfer{f_idx}));
        
        % Deblock the image horizontally
        deblock_h_all = deblock_mex(deblock_v_all')';
        deblock_h = double(deblock_h_all);
        
        % Update the results
        img_h_transfer{f_idx} = deblock_h;
        
    end
    
    other_info.runtime_deblock(f_idx) = toc;
end

end