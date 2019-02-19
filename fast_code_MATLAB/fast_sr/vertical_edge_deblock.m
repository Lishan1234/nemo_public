function img_deblock = vertical_edge_deblock(img, QP)
block_size = 8;
N_r = size(img, 1) / (block_size / 2);
N_c = size(img, 2) / block_size - 1;

[beta, tc] = get_beta_tc_from_QP(QP);

img_deblock = img;
for r_idx = 1:N_r
    for c_idx = 1:N_c
        r0 = 1 + (r_idx - 1) * (block_size / 2);
        c0 = (block_size / 2) + 1 + (c_idx - 1) * block_size;
        patch = img(r0:(r0 + block_size / 2 - 1), ...
            c0:(c0 + block_size - 1));
        
        % Do vertical filtering
        p_l1 = abs(patch(1, 2) - 2 * patch(1, 3) + patch(1, 4));
        q_l1 = abs(patch(1, 5) - 2 * patch(1, 6) + patch(1, 7));
        p_l4 = abs(patch(4, 2) - 2 * patch(4, 3) + patch(4, 4));
        q_l4 = abs(patch(4, 5) - 2 * patch(4, 6) + patch(4, 7));
        
        if p_l1 + q_l1 + p_l4 + q_l4 > beta
            % no deblocking!
            img_deblock(r0:(r0 + block_size / 2 - 1), ...
                c0:(c0 + block_size - 1)) = patch;
        else
            % judge strong or normal filtering
            b1 = (p_l1 + q_l1 < (beta / 8)) && ...
                (p_l4 + q_l4 < (beta / 8));
            b2 = abs(patch(1, 1) - patch(1, 4)) + ...
                abs(patch(1, 5) - patch(1, 8)) < (beta / 8);
            b3 = abs(patch(4, 1) - patch(4, 4)) + ...
                abs(patch(4, 5) - patch(4, 8)) < (beta / 8);
            b4 = abs(patch(1, 4) - patch(1, 5)) < 2.5 * tc;
            b5 = abs(patch(4, 4) - patch(4, 5)) < 2.5 * tc;
            
            if b1 && b2 && b3 && b4 && b5
                % strong deblocking filter
                for i = 1:4
                    patch(i, :) = strong_filter(patch(i, :), tc);
                end
            else
                % normal deblocking filter
                
                % Judge how many elements to change
                if (p_l1 + q_l1) < (beta / 8 * 3) && ...
                        (p_l4 + q_l4) < (beta / 8 * 3)
                    % Change p0, q0, p1, q1
                    for i = 1:4
                        patch(i, :) = normal_filter_p0q0p1q1(patch(i, :), tc);
                    end
                else
                    % Only change p0, q0
                    for i = 1:4
                        patch(i, :) = normal_filter_p0q0(patch(i, :), tc);
                    end
                end
            end
            img_deblock(r0:(r0 + block_size / 2 - 1), ...
                c0:(c0 + block_size - 1)) = patch;
        end
    end
end

end

function p_vec_deblock = strong_filter(p_vec, tc)
c = 2 * tc;
dp0 = floor((p_vec(2) + 2 * p_vec(3) - ...
    6 * p_vec(4) + 2 * p_vec(5) + p_vec(6) + 4) / 8);
dp1 = floor((p_vec(2) - 3 * p_vec(3) + p_vec(4) + p_vec(5) + 2) / 4);
dp2 = floor((2 * p_vec(1) - 5 * p_vec(2) + p_vec(3) + ...
    p_vec(4) + p_vec(5) + 4) / 8);

dq0 = floor((p_vec(7) + 2 * p_vec(6) - ...
    6 * p_vec(5) + 2 * p_vec(4) + p_vec(3) + 4) / 8);
dq1 = floor((p_vec(7) - 3 * p_vec(6) + p_vec(5) + p_vec(4) + 2) / 4);
dq2 = floor((2 * p_vec(8) - 5 * p_vec(7) + p_vec(6) + ...
    p_vec(5) + p_vec(4) + 4) / 8);


dp0 = min(max(-c, dp0), c);
dp1 = min(max(-c, dp1), c);
dp2 = min(max(-c, dp2), c);

dq0 = min(max(-c, dq0), c);
dq1 = min(max(-c, dq1), c);
dq2 = min(max(-c, dq2), c);

% fprintf('dp0 = %d, dp1 = %d, dp2 = %d, dq0 = %d, dq1 = %d, dq2 = %d\n', ...
%     dp0, dp1, dp2, dq0, dq1, dq2);

p_vec_deblock = p_vec + [0, dp2, dp1, dp0, dq0, dq1, dq2, 0];
p_vec_deblock = min(max(p_vec_deblock, 0), 255);
end

function p_vec_deblock = normal_filter_p0q0p1q1(p_vec, tc)

d0 = floor((9 * (p_vec(5) - p_vec(4)) - 3 * (p_vec(6) - p_vec(3)) + 8) / 16);
d0 = min(max(-tc, d0), tc);

dp1 = floor((floor((p_vec(2) + p_vec(4) + 1) / 2) - p_vec(3) + d0 + 1) / 2);
dq1 = floor((floor((p_vec(7) + p_vec(5) + 1) / 2) - p_vec(6) - d0 + 1) / 2);

dp1 = min(max(-tc, dp1), tc / 2);
dq1 = min(max(-tc, dq1), tc / 2);

% fprintf('p2 = %f, p3 = %f, p4 = %f, p5 = %f, p6 = %f, p7= %f, d0 = %f, dp1 = %f, dq1 = %f\n', ...
%     p_vec(2), p_vec(3), p_vec(4), p_vec(5), p_vec(6), p_vec(7), d0, dp1, dq1);

p_vec_deblock = p_vec;
p_vec_deblock(3:6) = p_vec_deblock(3:6) + [dp1, d0, -d0, dq1];
p_vec_deblock = min(max(p_vec_deblock, 0), 255);
end

function p_vec_deblock = normal_filter_p0q0(p_vec, tc)

d0 = floor((9 * (p_vec(5) - p_vec(4)) - 3 * (p_vec(6) - p_vec(3)) + 8) / 16);
d0 = min(max(-tc, d0), tc);

% fprintf('p3 = %f, p4 = %f, p5 = %f, p6 = %f, d0 = %f\n', p_vec(3), p_vec(4), ...
%     p_vec(5), p_vec(6), d0);

p_vec_deblock = p_vec;
p_vec_deblock(4:5) = p_vec_deblock(4:5) + [d0, -d0];
p_vec_deblock = min(max(p_vec_deblock, 0), 255);
end

function [beta, tc] = get_beta_tc_from_QP(QP)
% assert(QP == 37, ['For any other value other than 37, ' ...
%     'have not set how to determine beta, and tc']);
switch QP
    case 37
        beta = 40;
        tc = 5;
    case 42
        beta = 45;
tc = 8;
    case 47
        beta = 60;
        tc = 10;
    otherwise
        error('Unrecognized QP!');
end

end