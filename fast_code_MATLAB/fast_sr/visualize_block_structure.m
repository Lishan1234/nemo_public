function vis_img = visualize_block_structure(img, block_struct, plot_color)

if ~exist('plot_color', 'var')
    plot_color = 'r';
end

% hf = figure;


vis_img = img;

for i = 1:length(block_struct)
    x = block_struct(i).x;
    y = block_struct(i).y;
    w = block_struct(i).w;
    if isempty(x) || isempty(w)
        continue;
    end
    
    if isfield(block_struct, 'h')
        h = block_struct(i).h;
    else
        h = w;
    end
    
%     plot([x + 1, x + w, x + w, x + 1, x + 1], ...
%         [y + 1, y + 1, y + h, y + h, y + 1], ...
%         [plot_color, '-']);
    vis_img = plot_block_loc(vis_img, [h + 1, w + 1], [x + 1, y + 1], 0);
%     hold on;
end

% set(hf, 'Units', 'normalized', 'Position', [0, 0, 1, 1]);
% vis_img = frame2im(getframe(gca));
% close(hf);


end