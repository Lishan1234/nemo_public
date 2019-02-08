function img_with_blk = plot_block_loc(img, block_size, block_loc, stroke_w)
if ~exist('stroke_w', 'var')
    stroke_w = 2;
end

img_with_blk = img;

img_w = size(img, 2);
img_h = size(img, 1);

draw_color = 255 * [1, 0, 0];
xl1 = max(min(block_loc(1) - stroke_w, img_w), 1);
xl2 = max(min(block_loc(1) + stroke_w, img_w), 1);
xr1 = max(min(block_loc(1) + block_size(2) - 1 - stroke_w, img_w), 1);
xr2 = max(min(block_loc(1) + block_size(2) - 1 + stroke_w, img_w), 1);

yt1 = max(min(block_loc(2) - stroke_w, img_h), 1);
yt2 = max(min(block_loc(2) + stroke_w, img_h), 1);
yb1 = max(min(block_loc(2) + block_size(1) - 1 - stroke_w, img_h), 1);
yb2 = max(min(block_loc(2) + block_size(1) - 1 + stroke_w, img_h), 1);

for c = 1:3
    img_with_blk(yt1:yt2, xl1:xr2, c) = draw_color(c);
    img_with_blk(yb1:yb2, xl1:xr2, c) = draw_color(c);
    img_with_blk(yt1:yb2, xl1:xl2, c) = draw_color(c);
    img_with_blk(yt1:yb2, xr1:xr2, c) = draw_color(c);   
end
