function res_recon = reconstruct_res(res_struct, img_size)
res_recon = zeros(img_size);
hit_map = zeros(img_size);

for i = 1:length(res_struct)
    if isempty(res_struct(i).x) || isempty(res_struct(i).w) || ...
            isempty(res_struct(i).residual)
        continue;
    end
    x0 = res_struct(i).x;
    y0 = res_struct(i).y;
    r = y0 + 1;
    c = x0 + 1;
    w = res_struct(i).w;
    
    assert(all(all(hit_map(r:(r + w - 1), c:(c + w - 1)) == 0)), ...
        'duplicate coverage!');
    
    tmp = reshape(res_struct(i).residual, [w, w])';
    res_recon(r:(r + w - 1), c:(c + w - 1)) = tmp;
    hit_map(r:(r + w - 1), c:(c + w - 1)) = 1;
end

end