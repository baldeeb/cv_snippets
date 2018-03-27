% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);

S = exp(input); N = sum(S);
output = S ./ N;

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 
dv_input = [];

if backprop
%     TODO: Check the two websites below:
%       http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
%       https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
%       https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function

    intermediate = zeros(num_classes, num_classes);
    
    for b = 1:batch_size
        for i = 1:num_classes
            for j = 1:num_classes
                if i == j
                    intermediate(i, j) = output(i, b) * (1 - output(i, b));
                else
                    intermediate(i, j) = -output(i, b) * output(j, b);
                end
            end
        end
        dv_input(:, b) = intermediate * dv_output(:, b);
    end
end
