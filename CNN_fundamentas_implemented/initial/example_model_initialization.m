% Basic script to create a new network model

addpath layers;
addpath ..\data;

% l = [init_layer('conv',struct('filter_size',2,'filter_depth',3,'num_filters',2))
% 	init_layer('pool',struct('filter_size',2,'stride',2))
% 	init_layer('relu',[])
% 	init_layer('flatten',struct('num_dims',4))
% 	init_layer('linear',struct('num_in',32,'num_out',10))
% 	init_layer('softmax',[])];

% model = init_model(l,[10 10 3],10,true);

% Load input and label data
load_MNIST_data;


[in_height, in_width, in_depth, batch_size] =  size(train_data);
num_labels = 10;  % size(train_label, 2);
new_train_data = train_data(:, :, :, 1:batch_size/100);
new_label_data = train_label(1:batch_size/100);


l = [init_layer('conv',struct('filter_size',2,'filter_depth',in_depth,'num_filters',2))
	init_layer('pool',struct('filter_size',2,'stride',2))
	init_layer('relu',[])
	init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',338,'num_out',num_labels))
	init_layer('softmax',[])];

model = init_model(l, size(new_train_data), num_labels, true);

% Example calls you might make:
[output, activations] = inference(model,new_train_data);
[loss, dv_input] = loss_euclidean(output,new_label_data.',[],true);
[grad] = calc_gradient(model, new_train_data, activations, dv_input);

weight_update_hyper_params = ...
    struct('learning_rate', 0.05, 'weight_decay', 0.01);

[model] = update_weights(model, grad, weight_update_hyper_params);
