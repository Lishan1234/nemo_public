function recon_together = combine_dumped_info(intra_recon, ...
    inter_mc, res_all, inter_mask)

N_frames = length(intra_recon);
img_width = size(intra_recon{1}, 2);
img_height = size(intra_recon{1}, 1);

recon_together = cell(1, N_frames);

for f_idx = 1:N_frames
    recon_frame = zeros(img_height, img_width);
    intra_mask_frame = inter_mask{f_idx} == 0;
    recon_frame(intra_mask_frame) = intra_recon{f_idx}(intra_mask_frame);
    recon_frame(~intra_mask_frame) = inter_mc{f_idx}(...
        ~intra_mask_frame) + res_all{f_idx}(~intra_mask_frame);
    recon_together{f_idx} = recon_frame;
end

end