function enc_info = encode_sequence(seq_name, enc_params)

if ~exist('enc_params', 'var')
    enc_params = make_encoding_param();
end

% Find the exact file name
dir_ent = dir(fullfile(enc_params.test_yuv_dir, [seq_name, '*.yuv']));
if isempty(dir_ent)
    error('Could not find the file for the sequence!');
end
yuv_filename = dir_ent(1).name;
[~, img_width, img_height] = parse_test_yuv_name(yuv_filename);

% Input arguments
enc_app = fullfile(enc_params.hm_bin_dir, 'TAppEncoderStatic');
cfg_main = fullfile(enc_params.cfg_dir, ...
    [enc_params.main_cfg_setting, '.cfg']);
cfg_sequence = fullfile(enc_params.cfg_sequence_dir, [seq_name, '.cfg']);
input_yuv = fullfile(enc_params.test_yuv_dir, yuv_filename);

% Output arguments
yuv_recon_dir = fullfile(cd, 'enc_files');
if ~exist(yuv_recon_dir, 'dir')
    mkdir(yuv_recon_dir);
end

enc_alias = sprintf('%s_%d', seq_name, enc_params.num_frames);
yuv_recon_name = fullfile(yuv_recon_dir, [enc_alias, '.yuv']);
binary_name = fullfile(yuv_recon_dir, [enc_alias, '.str']);

% Other arguments
other_arguments = [];
if ~isempty(enc_params.num_frames)
    other_arguments = [other_arguments, ...
        sprintf('--FramesToBeEncoded=%d', enc_params.num_frames)];
end
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
    'dst_dir', {yuv_recon_dir}, ...
    'seq_alias', {enc_alias});

end