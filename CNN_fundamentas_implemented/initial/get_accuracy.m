function [accuracy] = get_accuracy(output, ground_truth)
    [~, pred_lables] = max(output, [], 1);
    accuracy = sum(pred_lables(:) == ground_truth(:))/size(ground_truth, 1);
end

