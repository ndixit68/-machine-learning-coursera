function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

% calculating preceision & recall for each epsilon

predictions = (pval < epsilon);
True_positives = sum(predictions(:)==1 & yval(:)==1); % positive here is that the item is an outlier
False_positives = sum(predictions(:)==1 & yval(:)==0); % algorithm predicted putlier but its non-outlier
True_negatives = sum(predictions(:)==0 & yval(:)==0); % algorith predicted non-outlier and it is actually a non-outlier
False_negatives = sum(predictions(:)==0 & yval(:)==1); % algorith predicted non-outlier but it is an outlier

Preceision = True_positives/(True_positives + False_positives);
Recall = True_positives/(True_positives + False_negatives);

F1 = 2*((Preceision*Recall)/(Preceision+Recall));
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
