function [ Error ] = Step2function_leaveoneout( DataOpen, DataClose, ...
  Classes, Parameters, n)
% Step3_leaveoneout performs leave-one-out cross-testing for values 
% starting from n

% If no "n" parameter then we will do cross testing for every subject
if (nargin < 5)  ||  isempty(n)
	n = 1;
end

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
Var = [0.7, 0.9, 0.99]; % Variance explained by PCA

% C parameters for SVM:
iC = Parameters(6);
C = [0.01, 1, 100];

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
[NumOfSamples, NumOfChannels, NumOfSpectrumPoints] = size(Data);
  
% If must take ROIs
if iRoi == 2

  %Temporary variable for single data
  DataTemp = zeros(NumOfSamples,7,NumOfSpectrumPoints);

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
  NumOfChannels = 7;

  % If using both data sets then second half need ROI also 
  if iDataType == 3
    
    % Temporary variable for data
    DataTemp=zeros(NumOfSamples,7,NumOfSpectrumPoints);

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
    NumOfChannels = 14;
  end
else
  Data1=Data;
end
    
clear('DataTemp');
clear('Data');

% All frequencies
if iFreq == 1

  % Initialize
  DataMatrix = zeros(NumOfSamples, NumOfChannels*NumOfSpectrumPoints);

  % For all samples
  for ii = 1:NumOfSamples
    count = 1;
    
    % For each channel
    for kk = 1:NumOfChannels;
      DataMatrix(ii, count:count+NumOfSpectrumPoints-1) = squeeze(Data1(ii, kk,:));
      count = count + NumOfSpectrumPoints;
    end
  end

% Bands based on power spectrum
elseif iFreq == 2

  % Initialize
  DataMatrix = zeros(NumOfSamples,NumOfChannels*99);

  for ii = 1:NumOfSamples
    count = 0;
    for kk = 1:NumOfChannels;
      for jj = 1:60
        count = count + 1;
        DataMatrix(ii,count) =  Data1(ii,kk,jj);
      end
      for jj = 61:2:99
        count = count + 1;
        DataMatrix(ii,count) = mean(Data1(ii,kk,jj:jj+1));
      end
      for jj = 101:4:145
        count = count + 1;
        DataMatrix(ii,count) = mean(Data1(ii,kk,jj:jj+3));
      end
      for jj = 149:40:389
        count = count + 1;
        DataMatrix(ii,count) = mean(Data1(ii,kk,jj:jj+39));
      end
    end
  end

% Bands based on literature  
else

  % Initialize
  DataMatrix = zeros(NumOfSamples,NumOfChannels*11);

  for ii = 1:NumOfSamples
    for jj = 1:NumOfChannels
      % "sub-delta"
      DataMatrix(ii,(jj-1)*11+1)=mean(Data1(ii,jj,2:5));
      % delta
      DataMatrix(ii,(jj-1)*11+2)=mean(Data1(ii,jj,6:17));
      % theta
      DataMatrix(ii,(jj-1)*11+3)=mean(Data1(ii,jj,18:33));
      % alpha
      DataMatrix(ii,(jj-1)*11+4)=mean(Data1(ii,jj,34:49));
      % low beta
      DataMatrix(ii,(jj-1)*11+5)=mean(Data1(ii,jj,50:81));
      % high beta
      DataMatrix(ii,(jj-1)*11+6)=mean(Data1(ii,jj,82:121));
      % lowlow gamma
      DataMatrix(ii,(jj-1)*11+7)=mean(Data1(ii,jj,122:161));
      % low gamma
      DataMatrix(ii,(jj-1)*11+8)=mean(Data1(ii,jj,162:200));
      % high gamma
      DataMatrix(ii,(jj-1)*11+9)=mean(Data1(ii,jj,201:279));
      % highhigh gamma
      DataMatrix(ii,(jj-1)*11+10)=mean(Data1(ii,jj,280:359));
      % superhigh gamma
      DataMatrix(ii,(jj-1)*11+11)=mean(Data1(ii,jj,360:497));
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

% Performing PCA
[~, Score, Eigen] = princomp(DataMatrix2,'econ');

% Calculating cumulative variance
CumEigen = cumsum(Eigen)/sum(Eigen);
        
% Choosing the number of dimensions by cumulative variance
MaxId = find(CumEigen>Var(iVar),1);

% Data after the PCA transformation
DataMatrix3 = Score(:,1:MaxId);
clear('DataMatrix2');

if iModel == 2
  
  % Finding RBF parameter for SVM with random classes
  ClassesRandInit = zeros(NumOfSamples,1);
  ClassesRandInit(1:ceil(NumOfSamples / 2)) = 1;
  ClassesRand = ClassesRandInit(randperm(NumOfSamples));

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
    ErrorTemp = sum(abs(RE - ClassesRand)) / NumOfSamples;
    
    % If needed error is reached, break and continue
    if  ErrorTemp > RBFerror(iRBFerror)
      RBFsigma = Values(jj);
      break
    end
  end
end

% Testing each subject from the test set
Correct = 0;

for iForTesting = n : NumOfSamples
  
  % Getting the data
  xtest = DataMatrix3(iForTesting,:);
  
  % Using full data, except the one which is being classified
  xtrain = DataMatrix3;
  xtrain(iForTesting,:) = [];
  
  ytrain = Classes;
  ytrain(iForTesting,:) = [];
  
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
  Correct = Correct + (Classified == Classes(iForTesting));

end

% Total error
TotelTest = NumOfSamples - n + 1;
Error = (TotelTest - Correct) / TotelTest;

end