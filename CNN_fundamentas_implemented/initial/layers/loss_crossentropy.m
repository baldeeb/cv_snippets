% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% labels: batch_size x 1
% ----------------------------------------------------------------------

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));

oneh_labels = zeros(size(input));
for i = 1:size(input, 2)
   oneh_labels(labels(i), i) = 1;
end

loss = -sum(sum(oneh_labels .* log(input))) ./ size(input, 2);

if backprop
    dv_input = -oneh_labels ./ input;
else
    dv_input = [];
end
