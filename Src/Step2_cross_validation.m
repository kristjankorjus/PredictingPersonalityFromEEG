function [ Correct ] = Step2_cross_validation( DataOpen, DataClose, ...
  Classes, Parameters, train_id, test_id)
% Step2_cross_validation performs cross-validation with a given set of the
% best hyper-parameters.

% Function performs a cross-validation step for the article 
% "Personality cannot be predicted from the power of resting state 1 EEG"
% submitted to Frontiers in Human Neuroscience
%
% Output: the number of correctly clasified subjects
%
% Tested with Windows 8 and Matlab 2013a with Statistics Toolbox
%
% Comments and questions: Kristjan Korjus (korjus@gmail.com)

%%%%%%%%%%%%%%%%%%%
% Parameter space %
%%%%%%%%%%%%%%%%%%%

% Type of the data:
iDataType = Parameters(1); % Eyes open, eyes close, open + close

% Different Regions of Interests (ROI):
iRoi = Parameters(2); % All, Literature

% Different frequency bands:
iFreq = Parameters(3); % All, Customized pooling, based on literature

% Normalization:
iNorm = Parameters(4); % Column, Row normalization, No normalization

% Choosing number of dimensions by total variance explained by PCA:
iVar = Parameters(5);
Var = [0.7, 0.9, NaN]; % Variance explained by PCA

% C parameters for SVM:
iC = Parameters(6);
C = [0.01, 100];

% Choosing the model:
iModel = Parameters(7); % Linear SVM, SVM with RBF kernel

% Finding RBF sigma parameter for SVM:
iRBFerror = Parameters(8);
RBFerror = [0.1, 0.3]; % RBF sigma parameter error rate for non-linear SVM


% Analysis

% Different data used
if iDataType == 1
  Data = DataOpen;
elseif iDataType == 2
  Data = DataClose;
else
  Data = [DataOpen,DataClose];
end
  
% Size of the data
[num_of_samples, num_of_channels, num_of_spectrum_points] = size(Data);

% If must take ROIs
if iRoi == 2

  %Temporary variable for single data
  DataTemp = zeros(num_of_samples,7,num_of_spectrum_points);

  % lFr = 'F3', 'F7', 'AF3', 'Fp1'
  DataTemp(:,1,:) = mean(Data(:,[4, 3, 2, 1],:),2);
  % rFr = 'F4', 'F8', 'AF4', 'Fp2
  DataTemp(:,2,:) = mean(Data(:,[27, 28, 29, 30],:),2);
  % lCnt = 'FC5', 'CP5', 'T7', 'P7', 'C3'
  DataTemp(:,3,:) = mean(Data(:,[6, 10, 7, 11, 8],:),2);
  % rCnt = 'FC6', 'CP6', 'T8', 'P8', 'C4'
  DataTemp(:,4,:) = mean(Data(:,[25, 21, 24, 20, 23],:),2);
  % mFr = 'Fz', 'Cz', 'FC1', 'FC2'
  DataTemp(:,5,:) = mean(Data(:,[31, 32, 5, 26],:),2);
  % mPar = 'Pz', 'CP1', 'CP2', 'P3', 'P4'
  DataTemp(:,6,:) = mean(Data(:,[13, 9, 22, 12, 19],:),2);
  % Occ = 'PO3' 'PO4', 'O1', 'Oz', 'O2'
  DataTemp(:,7,:) = mean(Data(:,[14, 18, 15, 16, 17],:),2);

  Data1 = DataTemp;

  %Changing the number of channels variable
  num_of_channels = 7;

  % If using both data sets then second half need ROI also 
  if iDataType == 3
    
    % Temporary variable for data
    DataTemp=zeros(num_of_samples,7,num_of_spectrum_points);

    % lFr = 'F3', 'F7', 'AF3', 'Fp1'
    DataTemp(:,1,:) = mean(Data(:,32+[4, 3, 2, 1],:),2);
    % rFr = 'F4', 'F8', 'AF4', 'Fp2
    DataTemp(:,2,:) = mean(Data(:,32+[27, 28, 29, 30],:),2);
    % lCnt = 'FC5', 'CP5', 'T7', 'P7', 'C3'
    DataTemp(:,3,:) = mean(Data(:,32+[6, 10, 7, 11, 8],:),2);
    % rCnt = 'FC6', 'CP6', 'T8', 'P8', 'C4'
    DataTemp(:,4,:) = mean(Data(:,32+[25, 21, 24, 20, 23],:),2);
    % mFr = 'Fz', 'Cz', 'FC1', 'FC2'
    DataTemp(:,5,:) = mean(Data(:,32+[31, 32, 5, 26],:),2);
    % mPar = 'Pz', 'CP1', 'CP2', 'P3', 'P4'
    DataTemp(:,6,:) = mean(Data(:,32+[13, 9, 22, 12, 19],:),2);
    % Occ = 'PO3' 'PO4', 'O1', 'Oz', 'O2'
    DataTemp(:,7,:) = mean(Data(:,32+[14, 18, 15, 16, 17],:),2);

    Data1 = [Data1, DataTemp];

    %Changing the number of channels variable
    num_of_channels = 14;
  end
else
  Data1=Data;
end
    
clear('DataTemp');
clear('Data');

% All frequencies
if iFreq == 1

  % Initialize
  DataMatrix = zeros(num_of_samples, num_of_channels*num_of_spectrum_points);

  % For all samples
  for ii = 1:num_of_samples
    count = 1;
    % For each channel
    for kk = 1:num_of_channels;
      DataMatrix(ii, count:count+num_of_spectrum_points-1) = squeeze(Data1(ii, kk,:));
      count = count + num_of_spectrum_points;
    end
  end

% Bands based on power spectrum
elseif iFreq == 2

  % Initialize
  DataMatrix = zeros(num_of_samples,num_of_channels*65);

  for ii = 1:num_of_samples
    count = 0;
    for kk = 1:num_of_channels;
      for jj = 1:50
        count = count + 1;
        DataMatrix(ii,count) =  Data1(ii,kk,jj);
      end
      for jj = 51:2:70
        count = count + 1;
        DataMatrix(ii,count) = mean(Data1(ii,kk,jj:jj+1));
      end
      for jj = 71:20:183-21
        count = count + 1;
        DataMatrix(ii,count) = mean(Data1(ii,kk,jj:jj+19));
      end
    end
  end

% Bands based on literature  
else

  % Initialize
  DataMatrix = zeros(num_of_samples,num_of_channels*10);

  for ii = 1:num_of_samples
    for jj = 1:num_of_channels
      % "sub-delta"
      DataMatrix(ii,(jj-1)*10+1)=mean(Data1(ii,jj,1));
      % delta
      DataMatrix(ii,(jj-1)*10+2)=mean(Data1(ii,jj,2:8));
      % theta
      DataMatrix(ii,(jj-1)*10+3)=mean(Data1(ii,jj,9:16));
      % alpha
      DataMatrix(ii,(jj-1)*10+4)=mean(Data1(ii,jj,17:24));
      % low beta
      DataMatrix(ii,(jj-1)*10+5)=mean(Data1(ii,jj,25:40));
      % high beta
      DataMatrix(ii,(jj-1)*10+6)=mean(Data1(ii,jj,41:60));
      % lowlow gamma
      DataMatrix(ii,(jj-1)*10+7)=mean(Data1(ii,jj,61:80));
      % low gamma
      DataMatrix(ii,(jj-1)*10+8)=mean(Data1(ii,jj,81:100));
      % high gamma
      DataMatrix(ii,(jj-1)*10+9)=mean(Data1(ii,jj,101:139));
      % highhigh gamma
      DataMatrix(ii,(jj-1)*10+10)=mean(Data1(ii,jj,140:183));
    end
  end
end


% If normalization needed then do it
if iNorm ~= 3
  DataMatrix2 = zscore(DataMatrix,[],iNorm);
else
  DataMatrix2 = DataMatrix;
end

clear('DataMatrix');

if ~isnan(Var(iVar))
  
  % Performing PCA
  [~, Score, Eigen] = princomp(DataMatrix2,'econ');

  % Calculating cumulative variance
  CumEigen = cumsum(Eigen)/sum(Eigen);

  % Choosing the number of dimensions by cumulative variance
  MaxId = find(CumEigen>Var(iVar),1);

  % Data after the PCA transformation
  DataMatrix3 = Score(:,1:MaxId);

else
  
  DataMatrix3 = DataMatrix2;
  
end

clear('DataMatrix2');

if iModel == 2
  
  % Finding RBF parameter for SVM with random classes
  ClassesRandInit = zeros(num_of_samples,1);
  ClassesRandInit(1:ceil(num_of_samples / 2)) = 1;
  ClassesRand = ClassesRandInit(randperm(num_of_samples));

  % If does not converge then sigma will be 100 001
  RBFsigma=100001;
  
  % Sigma values tried
  Values = [0.1.^(5:-1:1), 0.1:0.1:1, 2:1:20, 25:5:100, ...
    1000:1000:10000, 20000:10000:100000];
  
  % Try out increasing sigma values until error is reached
  for jj = 1:length(Values)
    
    % Train and classify same points
    RE = svmclassify(svmtrain(DataMatrix3,ClassesRand,...
      'kernel_function','rbf','rbf_sigma',Values(jj),...
      'boxconstraint',C(iC),'options',...
      optimset('maxiter',100000)),DataMatrix3);
    
    % Error
    ErrorTemp = sum(abs(RE - ClassesRand)) / num_of_samples;
    
    % If needed error is reached, break and continue
    if  ErrorTemp > RBFerror(iRBFerror)
      RBFsigma = Values(jj);
      break
    end
  end
end

 
% Getting the data
xtrain = DataMatrix3(train_id,:);
ytrain = Classes(train_id,:);

xtest = DataMatrix3(test_id,:);
ytest = Classes(test_id,:);

if iModel == 1
  % Linear SVM
  Classified = svmclassify(...
    svmtrain(xtrain,ytrain,'boxconstraint',C(iC),...
    'kktviolationlevel',0.01,...
    'options',optimset('maxiter',1000000)),xtest);

else
  % Non-linear SVM
  Classified = svmclassify(svmtrain(...
  xtrain,ytrain,'kernel_function','rbf',...
  'rbf_sigma',RBFsigma,'boxconstraint',C(iC),...
  'options',optimset('maxiter',1000000)),xtest);
end

% Adding if the result is correct
Correct = sum(Classified == ytest);

end