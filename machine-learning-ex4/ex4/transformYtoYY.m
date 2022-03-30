function yy = transformYtoYY(y, num_labels)
I = eye(num_labels);
yy = zeros(num_labels, length(y));

for index = 1:length(y)
 yy(:, index) = I(y(index),:)';
end

yy
end
