function params = make_encoding_param(varargin)

filepath = mfilename('fullpath');
[tool_folder_path, ~, ~] = fileparts(filepath);

% ------------------------------------------------------------------------
% Default parameters
% ------------------------------------------------------------------------
params.hm_main_dir = fullfile(tool_folder_path, 'hm'); 
params.hm_bin_dir = fullfile(params.hm_main_dir, 'bin');
params.cfg_dir = fullfile(params.hm_main_dir, 'cfg');
params.cfg_sequence_dir = fullfile(params.cfg_dir, 'per-sequence');
params.width = [];
params.height = [];
params.QP = [];

params.test_yuv_dir = fullfile(cd, 'data');

params.main_cfg_setting = 'encoder_lowdelay_P_main';

params.num_frames = [];

% ------------------------------------------------------------------------
% Parse the new parameters
% ------------------------------------------------------------------------
for i = 1:2:length(varargin)
    params.(varargin{i}) = varargin{i + 1};
end
