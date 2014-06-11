% Code file 2 for the article "Predicting personality from the resting 
% state EEG" submitted to Frontiers in Human Neuroscience
%
% Description:
%
% After finding the best combination of hyper-parameters in a previous step
% This script estimates the error rate on the other half of the subjects
% It also calles the function "Step2function_leaveoneout.m"
%
% Tested with Windows 8 and Matlab 2013a with Statistics Toolbox
%
% Comments and questions: Kristjan Korjus (korjus@gmail.com)

%% Init 
cd('CURRENT FOLDER');

%% First new data point which was not used in cross-validation
n = 131;

%% Load data
load('DataOpen.mat');
load('DataClose.mat');
load('Classes');
load('Results.mat');

%% Choosing the best model and performing leave-one-out cross-testing

Errors = zeros(1,5);

for ii = 1:5
  % Sorting the rows
  ResultsSorted = sortrows(Results, ii);
  
  % Keeping only the parameters
  Parameters = ResultsSorted(1,7:14);
  
  % Calculating the error
  Errors(ii) = Step2function_leaveoneout(DataOpen, DataClose, ...
    Classes(:,ii),Parameters,n);
end

%% Statistical significance
% Number of subjects in the test set
n2 = (388-n+1);

% P-values
p_valules = binocdf(Errors*n2,n2,0.5);
