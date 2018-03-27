function [test_loss, test_accuracy] = get_test_stats(model, test_batch_size)
    % Load input and label data if not loaded
    load_MNIST_data;
    
    % Randomly select batch 
    sample_indexes = randsample(size(test_data, 4), test_batch_size);
    in_data_seg = test_data(:,:,:, sample_indexes);
    in_label_seg = test_label(sample_indexes);
    
    [output, ~] = inference(model, in_data_seg);
%     [loss, dv_input] = loss_euclidean(output,in_label_seg,[],false);
    [test_loss, ~] = loss_crossentropy(output, in_label_seg, [], false);
    test_accuracy = get_accuracy(output, in_label_seg);
end

