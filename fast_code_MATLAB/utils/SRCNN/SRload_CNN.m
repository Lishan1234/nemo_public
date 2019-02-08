function Model = SRload_CNN()

file_path = mfilename('fullpath');
[pathstr, ~, ~] = fileparts(file_path);

x2_model = fullfile(pathstr, 'model', '9-5-5(ImageNet)', 'x2.mat');
x3_model = fullfile(pathstr, 'model', '9-5-5(ImageNet)', 'x3.mat');
x4_model = fullfile(pathstr, 'model', '9-5-5(ImageNet)', 'x4.mat');

Model = struct('x2_model', {x2_model}, 'x3_model', {x3_model}, ...
    'x4_model', {x4_model});


end