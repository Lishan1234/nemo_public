function res_h = Blockwise_upsample_with_TU(res_l, TU)

res_h = zeros(2 * size(res_l));

for i = 1:length(TU)
    if isempty(TU(i).x) || isempty(TU(i).w)
        continue;
    end
    
    x_l = TU(i).x;
    y_l = TU(i).y;
    x_h = 2 * x_l;
    y_h = 2 * y_l;
    
    w = TU(i).w;
    
    res_h((y_h + 1):(y_h + 2 * w), (x_h + 1):(x_h + 2 * w)) = ...
        imresize(res_l((y_l + 1):(y_l + w), (x_l + 1):(x_l + w))...
        , 2, 'bicubic');
end

end