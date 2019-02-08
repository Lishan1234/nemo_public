function [recon_intra, intra_mask] = get_recon_intra(PU_all, prefilt_Y)

N_frames = length(PU_all);
recon_intra = cell(1, N_frames);
intra_mask = cell(1, N_frames);

for t = 1:N_frames
    N_PU = length(PU_all{t});
    recon_intra{t} = zeros(size(prefilt_Y{1}, 1), size(prefilt_Y{1}, 2));
    intra_mask{t} = false(size(prefilt_Y{1}, 1), size(prefilt_Y{1}, 2));
    for pu_idx = 1:N_PU
        pu_struct = PU_all{t}(pu_idx);
        if isempty(pu_struct.intra)
            continue;
        end
        
        if pu_struct.intra == 0
            continue;
        end
        r = pu_struct.y + 1;
        c = pu_struct.x + 1;
        w = pu_struct.w;
        h = pu_struct.h;
        recon_intra{t}(r:(r + w - 1), c:(c + h - 1)) = ...
            prefilt_Y{t}(r:(r + w - 1), c:(c + h - 1));
        intra_mask{t}(r:(r + w - 1), c:(c + h - 1)) = true;
    end
end

end