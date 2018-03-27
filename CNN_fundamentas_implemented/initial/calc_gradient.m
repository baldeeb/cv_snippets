function [grad] = calc_gradient(model, input, activations, dv_output)
% Calculate the gradient at each layer, to do this you need dv_output
% determined by your loss function and the activations of each layer.
% The loop of this function will look very similar to the code from
% inference, just looping in reverse.

num_layers = numel(model.layers);
grad = cell(num_layers,1);

% TODO: Determine the gradient at each layer with weights to be updated
%
% [in_height, in_width, ~, ~] = size(input);
% 
% dv_input = ones(size(input)) .* 1/(in_height * in_width);
% output = input;

for l_idx = num_layers:-1:1
    lyr = model.layers(l_idx);
    
    if (l_idx-1) > 0
        activation = activations{l_idx - 1};
    else
        activation = input;
    end
    
    [~, dv_output, curr_grad] = lyr.fwd_fn(activation, lyr.params, ...
            lyr.hyper_params, true, dv_output);
    
    grad{l_idx} = curr_grad; 
end
