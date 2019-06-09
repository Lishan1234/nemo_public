function [seq_name, imgWidth, imgHeight] = parse_test_yuv_name(...
    input_filename)

param_str = '(?<name>\w+)_(?<width>\d+)x(?<height>\d+)';
param_str_2 = '(?<name>\w+)_(?<height>\d+)p';

tokenNames = regexp(input_filename, param_str, 'names');
if isempty(tokenNames)
    % 'filename_720p'.
    tokenNames = regexp(input_filename, param_str_2, 'names');
    imgHeight = str2num(tokenNames.height);
    imgWidth = imgHeight * 1280 / 720;
else
    imgWidth = str2num(tokenNames.width);
    imgHeight = str2num(tokenNames.height);
end
seq_name= tokenNames.name;

end