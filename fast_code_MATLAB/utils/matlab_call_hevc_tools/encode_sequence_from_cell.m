function enc_info = encode_sequence_from_cell(image_cells, seq_name, ...
    enc_params, enc_dir)

if ~isfield(enc_params, 'QP')
    enc_params.QP = 27;
end
% image_cells:RGB data

img_height = size(image_cells{1}, 1);
img_width = size(image_cells{1}, 2);
num_frames = length(image_cells);

yuv_name = sprintf('%s_%dx%d_%d', seq_name, img_width, img_height, ...
    num_frames);
yuv_filename = sprintf('%s.yuv', yuv_name);

if ~exist('enc_dir', 'var') || isempty(enc_dir)
    enc_dir = fullfile(cd, '..', 'temp_data', yuv_name);
end

if ~exist(enc_dir, 'dir')
    mkdir(enc_dir);
end

saveFileYuv(rgbcell2mov(image_cells), fullfile(enc_dir, yuv_filename), 1);

% Input arguments
if ismac
    enc_app = fullfile(enc_params.hm_bin_dir, 'TAppEncoder');
elseif isunix
    enc_app = fullfile(enc_params.hm_bin_dir, 'TAppEncoderStatic');
elseif ispc
    enc_app = fullfile(enc_params.hm_bin_dir, 'TAppEncoder.exe');
else
    error('The code can only run on Windows, Linux, Mac');
end

cfg_main = fullfile(enc_params.cfg_dir, ...
    [enc_params.main_cfg_setting, '.cfg']);
cfg_sequence = fullfile(enc_params.cfg_sequence_dir, [seq_name, '.cfg']);
input_yuv = fullfile(enc_dir, yuv_filename);

% Output arguments
yuv_recon_name = fullfile(enc_dir, [yuv_name, '_recon.yuv']);
binary_name = fullfile(enc_dir, [yuv_name, '.str']);

% Number of Frames
other_arguments = sprintf('--FramesToBeEncoded=%d', num_frames);

% Image Width
other_arguments = [other_arguments, ...
    sprintf(' --SourceWidth=%d --QP=%d', img_width, enc_params.QP)];

% Image Height
other_arguments = [other_arguments, ...
    sprintf(' --SourceHeight=%d', img_height)];

enc_command = sprintf('%s -c %s -c %s -i %s -o %s -b %s %s', ...
    enc_app, cfg_main, cfg_sequence, input_yuv, yuv_recon_name, ...
    binary_name, other_arguments);

% Evaluate the command!
system(enc_command);

% Information about the encoding!
enc_info = struct('yuv_recon_name', {yuv_recon_name}, ...
    'binary_name', {binary_name}, ...
    'img_width', {img_width}, ...
    'img_height', {img_height}, ...
    'dst_dir', {enc_dir}, ...
    'seq_alias', {yuv_name});

end