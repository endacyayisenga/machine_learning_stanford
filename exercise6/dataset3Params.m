function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Initialization
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = C;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
min_err = 1;
best_C = 0;
best_sigma = 0;
for ii = 1 : length(C)
    for jj = 1 : length(sigma)
        model= svmTrain(X, y, C(ii), @(x1, x2) gaussianKernel(x1, x2, sigma(jj))); 
        pred = svmPredict(model, Xval);
        err = mean(double(pred ~= yval));
        if (err < min_err)
            min_err = err;
            best_C = C(ii);
            best_sigma = sigma(jj);
        end
    end
end

C = best_C;
sigma = best_sigma;


% =========================================================================

end
