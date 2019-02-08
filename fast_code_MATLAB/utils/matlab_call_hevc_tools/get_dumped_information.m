function dec_info = get_dumped_information(enc_params, enc_info)
% Decode the sequence and get the dumped information

if ismac
    env_cmd = 'env PRINT_COEFF=1 PRINT_INTRA=1 PRINT_MV=1 SAVE_PREFILT=1';
    enc_app = fullfile(enc_params.hm_bin_dir, 'TAppDecoder');
elseif isunix
    env_cmd = 'env PRINT_COEFF=1 PRINT_INTRA=1 PRINT_MV=1 SAVE_PREFILT=1';
    enc_app = fullfile(enc_params.hm_bin_dir, 'TAppDecoderStatic');
elseif ispc
    env_cmd = ['set PRINT_COEFF=1 & set PRINT_INTRA=1', ...
        ' & set PRINT_MV=1 & set SAVE_PREFILT=1 &'];
    enc_app = fullfile(enc_params.hm_bin_dir, 'TAppDecoder.exe');
else
    error('The code can only run on Windows, Linux, Mac');
end

binary_file = enc_info.binary_name;
yuv_recon_file = enc_info.yuv_recon_name;
dump_txt_name = fullfile(enc_info.dst_dir, [enc_info.seq_alias, '_dump.txt']);

dump_cmd = sprintf('%s %s -b %s -o %s > %s', env_cmd, enc_app, ...
    binary_file, yuv_recon_file, dump_txt_name);
system(dump_cmd);
dec_info = struct('dump_txt_name', {dump_txt_name});
dec_info.enc_info = enc_info;
end