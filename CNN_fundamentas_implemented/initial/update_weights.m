function [updated_model] = update_weights(model,grad,hyper_params)

num_layers = length(grad);
a = hyper_params.learning_rate;
lmda = hyper_params.weight_decay;
updated_model = model;

friction = hyper_params.friction;

for l_itr = 1:num_layers
    % Reference for momentum
    % http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf
    
    layer = updated_model.layers(l_itr).params;
    prv_gw = updated_model.layers(l_itr).params.grad.W;
    prv_gb = updated_model.layers(l_itr).params.grad.b;
	
    % With momentum
    velW = friction * prv_gw + grad{l_itr}.W;
    velb = friction * prv_gb + grad{l_itr}.b;
%     
%     % No momentum
%     velW = grad{l_itr}.W;
%     velb = grad{l_itr}.b;
%     
    updated_model.layers(l_itr).params.W = layer.W - (velW .* a);  % - (layer.W .* lmda);
    updated_model.layers(l_itr).params.b = layer.b - (velb .* a);%  - (grad{l_itr}.b .* a) - (layer.b .* lmda);
	
	% Update gradient
	 updated_model.layers(l_itr).params.grad = grad{l_itr};
	
end




% 
% 
% 
% function updated_model = update_weights(model,grad,hyper_params)
% 
% num_layers = length(grad);
% a = hyper_params.learning_rate;
% lmda = hyper_params.weight_decay;
% updated_model = model;
% 
% % TODO: try apply momentum
% 
% for l_itr = 1:num_layers
%     layer = updated_model.layers(l_itr).params;
%     updated_model.layers(l_itr).params.W = ...
%         layer.W - (grad{l_itr}.W .* a) - (layer.W .* lmda);
%     updated_model.layers(l_itr).params.b = ...
%         layer.b - (grad{l_itr}.b .* a) - (layer.b .* lmda);
% end
