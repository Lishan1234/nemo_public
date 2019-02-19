function imh = SR_CNN(iml, ratio, model)

switch ratio
    case 2
        sr_model = model.x2_model;
    case 3
        sr_model = model.x3_model;
    case 4
        sr_model = model.x4_model;
    otherwise
        error('Unsupported upsampling ratio: %f', ratio);
end

iml = single(iml) / 255;
imh_bicubic = imresize(iml, ratio, 'bicubic');
imh_sr = SRCNN(sr_model, imh_bicubic);


border = ratio;
imh = imh_bicubic;
imh((1 + border):(end - border), (1 + border):(end - border)) = ...
    imh_sr((1 + border):(end - border), (1 + border):(end - border));

imh = uint8(255 * imh);
end