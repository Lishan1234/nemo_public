function res_h = Blockwise_upsample_with_TU(res_l, TU, sr_ratio)

res_h = zeros(sr_ratio * size(res_l));

for i = 1:length(TU)
    if isempty(TU(i).x) || isempty(TU(i).w)
        continue;
    end
    
    x_l = TU(i).x;
    y_l = TU(i).y;
    x_h = sr_ratio * x_l;
    y_h = sr_ratio * y_l;
    
    w = TU(i).w;
    
    res_h((y_h + 1):(y_h + sr_ratio * w), (x_h + 1):(x_h + sr_ratio * w)) = ...
        imresize(res_l((y_l + 1):(y_l + w), (x_l + 1):(x_l + w))...
        , sr_ratio, 'bicubic');
end

end