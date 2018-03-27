% Basic script to create a new network model

addpath layers;
addpath pcode;
addpath ..\data;
% Reference: 
% https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/


% Load input and label data
load_MNIST_data;


[in_height, in_width, in_depth, batch_size] =  size(train_data);

num_labels = 10;

l = [init_layer('conv',struct('filter_size',5,'filter_depth',in_depth,'num_filters',20))
	init_layer('relu',[])
	init_layer('pool',struct('filter_size',2,'stride',2))
    init_layer('conv',struct('filter_size',5,'filter_depth',20,'num_filters',50))
	init_layer('relu',[])
	init_layer('pool',struct('filter_size',2,'stride',2))
	init_layer('flatten',struct('num_dims',4))
	init_layer('relu',[])	
	init_layer('linear',struct('num_in',800,'num_out',500))
	init_layer('relu',[])	
	init_layer('linear',struct('num_in',500,'num_out', num_labels))
	init_layer('softmax',[])];

model = init_model(l, [in_height, in_width, in_depth], num_labels, true);

num_big_loops = 1 6;
num_train_loops = 30;

prog_bar = waitbar(0,'Initializing waitbar...');
waitbar(0, prog_bar, sprintf('%.0d%% Complete Progress...', 0));

train_stats = [];
test_stats = zeros(num_big_loops, 2);

% params initialization 
params = struct('learning_rate',0.07,'weight_decay',0.0005, ...
    'batch_size', 128, 'friction', 0.5);
l_rate_decay = 0.0025;

for i = 1:num_big_loops
    
    % Adjust ratio
    if i == 2
        params.friction = 0.95;
    elseif i >= 10
         params.learning_rate = params.learning_rate - l_rate_decay *(i-9);  % 0.05;
    end
    
    [model, loss]  = train(model, train_data, train_label, params, num_train_loops);

    % Calculate and store test and training loss every save_itr
    % iterations
    [test_stats(i, 1), test_stats(i, 2)] = get_test_stats(model, 800);
    train_stats = [train_stats; loss];

    % Save model
%         file_name = strcat('model', '_itr',num2str(i),'_', '');
%         save(file_name, 'model');        

    % Print i to show progress
    perc = i/num_big_loops;
    waitbar(perc, prog_bar, sprintf('%.0d%% Complete Progress...', perc*100));
end

close(prog_bar);

figure;
plot(train_stats(:, 1));
title('train loss');

figure;
plot(train_stats(:, 2));
title('train accuracy');

figure;
plot(test_stats(:, 1));
title('test loss');

figure;
plot(test_stats(:, 2));
title('test accurcy');
