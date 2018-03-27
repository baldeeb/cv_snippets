% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, padding for further work)
% params.W: filter_height x filter_width x filter_depth x num_filters
% params.b: num_filters x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)

[~,~,num_channels,batch_size] = size(input);
[filter_side,~,filter_depth,num_filters] = size(params.W);
assert(filter_depth == num_channels, 'Filter depth does not match number of input channels');

in_size = size(input);
out_height = size(input,1) - size(params.W,1) + 1;
out_width = size(input,2) - size(params.W,2) + 1;
output = zeros(out_height,out_width,num_filters,batch_size);

intermediate = zeros(out_height, out_width, num_channels);
for batch_idx = 1:batch_size
    for filter_idx = 1:num_filters
        for channel_idx = 1:filter_depth
            intermediate(:, :, channel_idx) = ...
                conv2( ...
                    input(:, :, channel_idx, batch_idx), ...
                    params.W(:, :, channel_idx, filter_idx), ... 
                    'valid' ... 
                );
        end
        
        output(:, :, filter_idx, batch_idx) =  ...
            sum(intermediate, 3) + params.b(filter_idx);
    end
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
%     Cool blog post below
%     https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e


    
    % Backpropagated error
    dv_input = zeros(size(input));
    intermediate = zeros(in_size(1), in_size(2), num_filters);
    for batch_idx = 1:batch_size
        for channel_idx = 1:filter_depth
            for filter_idx = 1:num_filters
                intermediate(:, :, filter_idx) = ...
                    conv2( ...
                        rot90( ... 
                        	params.W(:, :, channel_idx, filter_idx), ...
                        2), ... 
                        dv_output(:, :, filter_idx, batch_idx), ...
                        'full' ... 
                    );
            end
            dv_input(:, :, channel_idx, batch_idx) =  ...
                sum(intermediate, 3);
        end
    end


    % Calculate gradient
    grad.W = zeros(size(params.W));
    for filter_idx = 1:num_filters
        for channel_idx = 1:filter_depth
            for batch_idx = 1:batch_size
                grad.W(:, :, channel_idx,  filter_idx) =  ...
                    grad.W(:, :, channel_idx,  filter_idx) + ... 
                    conv2( ...
                        rot90( ...
                            input(:, :, channel_idx, batch_idx), ... 
                        2), ...
                        dv_output(:, :, filter_idx, batch_idx), ...
                        'valid' ... 
                    );
            end
        end
    end
    grad.W = grad.W ./ batch_size;
    
    grad.b = zeros(num_filters);
    for filter_idx = 1:num_filters
            grad.b(filter_idx) = ...
                sum(sum(sum(dv_output(:, :, filter_idx, :))));
%                 sum(reshape(dv_output(:, :, filter_idx, :),1,[]));
    end
    grad.b = grad.b(:, 1) ./ batch_size;
end
