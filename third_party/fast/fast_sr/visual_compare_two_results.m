function [cmp_image, block_mark_img, other_info] = visual_compare_two_results(m1_result, ...
    m2_result, ground_truth, params)
if ~exist('params', 'var')
    params = [];
end

if ~isfield(params, 'block_size')
    params.block_size = 32;
end

if ~isfield(params, 'large_block_size')
    large_block_size = floor(100 / params.block_size) * params.block_size;
    params.large_block_size = large_block_size;
end

if ~isfield(params, 'interp_method')
    params.interp_method = 'bilinear';
end

if ~isfield(params, 'max_block_num')
    params.max_block_num = 20;
end

if ~isfield(params, 'block_contrast_normalize')
    params.block_contrast_normalize = 0;
end

diff_1 = conv2((ground_truth - m1_result).^2, ones(params.block_size), ...
    'valid');
diff_2 = conv2((ground_truth - m2_result).^2, ones(params.block_size), ...
    'valid');

m1_gain = max(diff_2 - diff_1, 0);
m2_gain = max(diff_1 - diff_2, 0);

% ------------------------------------------------------------------------
% Pick the positive patches!
% ------------------------------------------------------------------------
pos_patches = cell(params.max_block_num, 3);
pos_loc = zeros(2, params.max_block_num);
for i = 1:params.max_block_num
    [v, idx] = max(m1_gain(:));
    [r, c] = ind2sub(size(m1_gain), idx);
    pos_patches{i, 1} = m1_result(r:(r + params.block_size - 1), ...
        c:(c + params.block_size - 1));
    pos_patches{i, 2} = m2_result(r:(r + params.block_size - 1), ...
        c:(c + params.block_size - 1));
    pos_patches{i, 3} = ground_truth(r:(r + params.block_size - 1), ...
        c:(c + params.block_size - 1));
    
    pos_loc(:, i) = [c; r];
    % Deactive the areas near [r, c]
    r0 = max(1, r - 2 * params.block_size);
    r1 = min(size(diff_1, 1), r + 2 * params.block_size);
    c0 = max(1, c - 2 * params.block_size);
    c1 = min(size(diff_1, 2), c + 2 * params.block_size);
    m1_gain(r0:r1, c0:c1) = 0;
end

% ------------------------------------------------------------------------
% Pick the negative patches!
% ------------------------------------------------------------------------
neg_patches = cell(params.max_block_num, 3);
neg_loc = zeros(2, params.max_block_num);
for i = 1:params.max_block_num
    [~, idx] = max(m2_gain(:));
    [r, c] = ind2sub(size(diff_1), idx);
    neg_patches{i, 1} = m1_result(r:(r + params.block_size - 1), ...
        c:(c + params.block_size - 1));
    neg_patches{i, 2} = m2_result(r:(r + params.block_size - 1), ...
        c:(c + params.block_size - 1));
    neg_patches{i, 3} = ground_truth(r:(r + params.block_size - 1), ...
        c:(c + params.block_size - 1));
    
    neg_loc(:, i) = [c; r];
    % Deactive the areas near [r, c]
    r0 = max(1, r - 2 * params.block_size);
    r1 = min(size(diff_1, 1), r + 2 * params.block_size);
    c0 = max(1, c - 2 * params.block_size);
    c1 = min(size(diff_1, 2), c + 2 * params.block_size);
    m2_gain(r0:r1, c0:c1) = 0;
end

% ------------------------------------------------------------------------
% Concatenate pos and neg patches!
% ------------------------------------------------------------------------
all_patches = [pos_patches, neg_patches];
for r = 1:size(all_patches, 1)
    for c = 1:size(all_patches, 2)
        if params.block_contrast_normalize == 1
            all_patches{r, c} = histeq(uint8(all_patches{r, c}));
        end
        all_patches{r, c} = imresize(all_patches{r, c}, ...
            [params.large_block_size, params.large_block_size], ...
            params.interp_method);
    end
end

h = figure;
imshow(uint8(ground_truth));
hold on;
bs = params.block_size;
for i = 1:params.max_block_num
    plot(pos_loc(1, i) + [0, bs - 1, bs - 1, 0, 0], ...
        pos_loc(2, i) + [0, 0, bs - 1, bs - 1, 0], 'r-');
    hold on;
end

for i = 1:params.max_block_num
    plot(neg_loc(1, i) + [0, bs - 1, bs - 1, 0, 0], ...
        neg_loc(2, i) + [0, 0, bs - 1, bs - 1, 0], 'b-');
    hold on;
end
block_mark_img = frame2im(getframe(gca));

close(h);

other_info = struct('pos_loc', {pos_loc}, 'neg_loc', {neg_loc});

cmp_image = cell2mat(all_patches);