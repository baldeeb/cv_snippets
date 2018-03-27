function [model, loss] = train(model,input,label,params,numIters)

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% Learning rate
if isfield(params,'learning_rate') lr = params.learning_rate;
else lr = .01; end
% Weight decay
if isfield(params,'weight_decay') wd = params.weight_decay;
else wd = .0005; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 128; end
% Momentum friction 
if isfield(params,'friction') friction = params.friction ;
else friction  = 0.9; end


% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd,...
    'friction', friction);

prog_bar = waitbar(0,'Initializing waitbar...');
waitbar(0, prog_bar, sprintf('%d%% Train Function Progress... itr:%d%', 0, 0));

tic 

loss = zeros(numIters, 2);
for inner_i = 1:numIters
    
    % Randomly select batch 
    sample_indexes = randsample(size(input, 4), batch_size);
    in_data_seg = input(:,:,:, sample_indexes);
    in_label_seg = label(sample_indexes);
    
    [output, activations] = inference(model, in_data_seg);
    [train_loss, dv_input] = loss_crossentropy(output,in_label_seg,[],true);
    [grad] = calc_gradient(model, in_data_seg, activations, dv_input);
    [model] = update_weights(model, grad, update_params);
    
    % Calculate training accuracy
    train_acc = get_accuracy(output, in_label_seg);
    
    % Save training loss and accuracy 
    loss(inner_i, :) = [train_loss, train_acc];
    
    % Display training loss and accuracy
    disp(strcat('Test: loss->', num2str(train_loss), ...
        '  accuracy->', num2str(train_acc), '.'));
    
    % Print i to show progress
    perc = inner_i/numIters;
    waitbar(perc, prog_bar, ...
        sprintf('%d%% Train Function Progress... itr:%d%',...
        perc*100, inner_i));
    
    toc
end
close(prog_bar);