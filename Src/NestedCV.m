
% This script performs a nested cross-validation analysis for the article 
% "Personality cannot be predicted from the power of resting state EEG"
% submitted to Frontiers in Human Neuroscience

% First, it calls a function Step1_nested_cross_validation.m, which chooses
% the best hyper-parameters

% Second, it predicts personality scores for each personality trait using
% the function Step2_cross_validation.m

% Third, it performs statistical significance analysis

% Tested with Windows 8 and Matlab 2013a with Statistics Toolbox
% Comments and questions: Kristjan Korjus (korjus@gmail.com)

%% Init

% Load data
load('../Data/Classes');
load('../Data/DataOpen');
load('../Data/DataClose');

% Fixed parameters
k_fold_parameter = 10; % k-fold parameter in cross-validation
nested_k_fold_parameter = 10; % k-fold parameter in nested-cross-validation


%% Cross-Validation partitions

% Random Partitions for the first cross-validation
partitions = crossvalind('Kfold', size(DataOpen,1), k_fold_parameter);

% Will save number of correct predictions for each personality score
Correct = zeros(1,5);

for ii = 1:k_fold_parameter
  
  % Indeces for test and train
  test_id = (partitions == ii);
  train_id = ~test_id;
    
  % Nested cross-validation for choosing the best parameters
  % Note: only train_id goes in
  
  BestParameters = Step1_nested_cross_validation(DataOpen(train_id,:,:), ...
    DataOpen(train_id,:,:), Classes(train_id,:), nested_k_fold_parameter,ii);
  
  % Will use the best parameters to classify test_id subjects   
  for jj = 1:5 % All five personality traits
    
    % Keeping only the parameters
    Parameters = BestParameters(jj,7:14);

    % Adding the number of correct predictions
    Correct(jj) = Correct(jj) + Step2_cross_validation(DataOpen, ...
    DataClose, Classes(:,jj), Parameters, train_id, test_id);
  
  end
end

%% Statistical significance - using binomial test

% Number of samples
n=309;

% Errors
Errors = (n - CorrectSum) / n;

% P-values
p_valules = binocdf(n-CorrectSum,n,0.5);
