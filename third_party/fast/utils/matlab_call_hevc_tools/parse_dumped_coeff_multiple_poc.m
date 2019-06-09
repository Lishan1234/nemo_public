function [PU_all, res_luma_all] = parse_dumped_coeff_multiple_poc(file_path)

% Parse Multiple Frames. Parse Intra and MV.

fid = fopen(file_path, 'r');
line = fgetl(fid);
PU_all = {};
res_luma_all = {};

tid = -1;

res_frame_all = cell(1, 3);
res_frame_all{1} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
res_frame_all{2} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
res_frame_all{3} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
PU_frame_all = struct('intra', {[]}, 'x', {[]}, 'y', {[]}, ...
    'w', {[]}, 'h', {[]}, ...
    'mv_x', {[]}, 'mv_y', {[]}, 't', {[]}, 't_r', {[]}, ...
    'luma_mode', {[]});

while ischar(line)
    % Judge whether the line starts with one of the keywords.
    line_segments = strsplit(line, {':', ',', ' '});
    switch lower(line_segments{1})
        case 'coeff'
            nums = cellstr2num(line_segments(2:end));
            qper = nums(1);
            qrem = nums(2);
            w = nums(3);
            list = nums(4);
            n = w * w;
            if length(nums) ~= 4 + 4 * n
                fprintf('Last line reached!');
            end
            scaling = nums((4 + 1):(4 + n));
            levels = nums((4 + n + 1):(4 + 2 * n));
            try
                coeffs = nums((4 + 2 * n + 1):(4 + 2 * n + n));
                residuals = nums((4 + 2 * n + n + 1):...
                (4 + 2 * n + n + n));
            catch
                db_var = 1;
            end
            
            
            res_frame_all{tid}(end).w = w;
            res_frame_all{tid}(end).residual = residuals;
            if all(levels == 0)
                %                 fprintf('Encountering an all-zero TU!\n');
                assert(all(coeffs == 0) && all(residuals == 0), ...
                    'A TU with all zero scalings has non zero coeffs & residuals');
            end
            tid = -1;
            %             fprintf('Parse a TU coeff of size %dx%d!\n', w, w);
        case 'tu'
            nums = cellstr2num(line_segments(2:end));
            x = nums(1);
            y = nums(2);
            text = nums(3);
            intra = nums(4);
            
            %             fprintf('Locate a TU at (%d, %d, %d, %d)!\n', ...
            %                 x, y, text, intra);
            
            % Initialize a TU
            res_struct = struct('x', {x}, 'y', {y}, 'w', {[]}, ...
                'residual', {[]});
            tid = max(text, 1);
            res_frame_all{tid}(length(res_frame_all{tid}) + 1) = res_struct;
        case 'intra'
            prop_map = parse_prop(line_segments(2:end));
            PU_frame = struct('intra', {1}, 'x', {prop_map('x')}, ...
                'y', {prop_map('y')}, ...
                'w', {prop_map('w')}, 'h', {prop_map('h')}, ...
                'mv_x', {[]}, 'mv_y', {[]}, 't', {prop_map('t')}, ...
                't_r', {[]}, 'luma_mode', {prop_map('luma_mode')});
            PU_frame_all(length(PU_frame_all) + 1) = PU_frame;
        case 'mv'
            prop_map = parse_prop(line_segments(2:end));
            PU_frame = struct('intra', {0}, 'x', {prop_map('x')}, ...
                'y', {prop_map('y')}, ...
                'w', {prop_map('w')}, 'h', {prop_map('h')}, ...
                'mv_x', {prop_map('mv_x')}, 'mv_y', {prop_map('mv_y')},...
                't', {prop_map('t')}, 't_r', {prop_map('t_r')}, ...
                'luma_mode', {[]});
            PU_frame_all(length(PU_frame_all) + 1) = PU_frame;
        case 'poc'
            fprintf('POC found! Start a new frame!\n');
            res_luma_all{length(res_luma_all) + 1} = res_frame_all{1};
            PU_all{length(PU_all) + 1} = PU_frame_all;
            res_frame_all = cell(1, 3);
            res_frame_all{1} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
            res_frame_all{2} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
            res_frame_all{3} = struct('x', {[]}, 'y', {[]}, 'w', {[]}, 'residual', {[]});
            PU_frame_all = struct('intra', {[]}, 'x', {[]}, 'y', {[]}, ...
                'w', {[]}, 'h', {[]}, ...
                'mv_x', {[]}, 'mv_y', {[]}, 't', {[]}, 't_r', {[]}, ...
                'luma_mode', {[]});
        otherwise
            fprintf('Unparsed statement: %s\n', line);
    end
    line = fgetl(fid);
end

fclose(fid);

end

function prop_map = parse_prop(line)
prop_map = containers.Map();
for i = 1:length(line)
    vars = strsplit(line{i}, '=');
    for t = 1:(length(vars) - 1)
        prop_map(vars{t}) = str2num(vars{end});
    end 
end

end