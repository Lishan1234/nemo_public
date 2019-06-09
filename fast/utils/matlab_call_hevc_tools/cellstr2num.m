function nums = cellstr2num(cell_str)

tmp = [cell_str(:)'; repmat({' '}, 1, length(cell_str))];
tmp = cell2mat(tmp(:)');
nums = str2num(tmp);

end