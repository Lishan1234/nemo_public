function mov = rgbcell2mov(image_cell)
N_frames = length(image_cell);
for i = 1:N_frames
    mov(i) = im2frame(image_cell{i});
end

end