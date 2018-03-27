function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

backprop = false;
dv_input = [];
output = input;

for l_idx = 1:num_layers
    lyr = model.layers(l_idx);
    [output, ~, ~] = ...
        feval(lyr.fwd_fn, output, lyr.params, lyr.hyper_params, 0, dv_input);
    
    activations{l_idx} = output;
end



