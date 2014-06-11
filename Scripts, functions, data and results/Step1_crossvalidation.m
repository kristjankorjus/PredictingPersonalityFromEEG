% Code file 1 for the article "Predicting personality from the resting
% state EEG" submitted to Frontiers in Human Neuroscience
%
% Description:
%
% This script loads the data, takes a sub-sample of it
% (size of the sub-sample = NUM_OF_SAMPLES) and runs k-fold
% cross-validation (k = K_FOLD) with the data.
% After finding the best combination of hyper-parameters next step
% "crossvalidation.m" should be called.
%
% Tested with Windows 8 and Matlab 2013a with Statistics Toolbox
%
% Comments and questions: Kristjan Korjus (korjus@gmail.com)

%% Initialize
cd('CURRENT FOLDER');

%% Finding best hyper parameters with k-fold cross-validation

% Fixed parameters
K_FOLD = 10; % k-fold parameter
NUM_OF_SAMPLES = 130; % Number of samples used in CV phase

% Load classes
load('Classes');

% Use NUM_OF_SAMPLES of the data for CV
Classes = Classes(1:NUM_OF_SAMPLES,:);

% Create partitions for CV
CVpartitions = cvpartition(NUM_OF_SAMPLES,'kfold',K_FOLD);
% PS. It would be possible to use the partition save to the current folder

%%%%%%%%%%%%%%%%%%%
% Parameter space %
%%%%%%%%%%%%%%%%%%%

% Using different types of data:
DataType = 1:3; % Eyes open, eyes close, open + close

% Different Regions of Interests (ROI):
Roi = 1:2; % All, Literature

% Different frequency bands:
Freq = 1:3; % All, customized pooling, based on literature

% Normalization:
Norm = 1:3; % Column normalization, Row normalization, No normalization

% Choosing number of dimensions by total variance explained by PCA:
Var = [0.7, 0.9, 0.99]; % Variance explained by PCA

% C parameters for SVM
C = [0.01, 1, 100];

% Choosing the model:
Model = 1:2; % Linear SVM, SVM with RBF kernel

% Finding RBF sigma parameter for SVM:
RBFerror = [0.1, 0.3]; % RBF sigma parameter error rate for non-linear SVM

ParemeterSpace = length(DataType)*length(Roi)*length(Freq)*length(Norm)*...
  length(Var)*length(C)*3;

%% Master for-loop

% Results
Results=[];

% Different data used
for iDataType = DataType
  
  if iDataType == 1
    load('DataOpen.mat');
    Data = DataOpen(1:NUM_OF_SAMPLES,:,:);
  elseif iDataType == 2
    load('DataClose.mat');
    Data = DataClose(1:NUM_OF_SAMPLES,:,:);
  else
    Data = [DataOpen(1:NUM_OF_SAMPLES,:,:),DataClose(1:NUM_OF_SAMPLES,:,:)];
    clear('DataOpen');
    clear('DataClose');
  end
  
  % Size of the data
  [NUM_OF_SAMPLES, NumOfChannels, NumOfSpectrumPoints] = size(Data);
  
  % Different Regions of Interests
  for iRoi = Roi

    % If must take ROIs
    if iRoi == 2
      
      %Temporary variable for single data
      DataTemp = zeros(NUM_OF_SAMPLES,7,NumOfSpectrumPoints);
      
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
      
      % If using both data sets then second half needs ROI also 
      if iDataType == 3
        
        %Temporary variable for data
        DataTemp=zeros(NUM_OF_SAMPLES,7,NumOfSpectrumPoints);
    
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

    % Different frequency bands (will transform 3D matrix to 2D)
    for iFreq = Freq

      % All frequencies
      if iFreq == 1
        
        % Initialize
        DataMatrix = zeros(NUM_OF_SAMPLES, NumOfChannels*NumOfSpectrumPoints);
        
        % For all samples
        for ii = 1:NUM_OF_SAMPLES
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
        DataMatrix = zeros(NUM_OF_SAMPLES,NumOfChannels*99);

        for ii = 1:NUM_OF_SAMPLES
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
        DataMatrix = zeros(NUM_OF_SAMPLES,NumOfChannels*11);

        for ii = 1:NUM_OF_SAMPLES
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

      % Normalization
      for iNorm = Norm
        
        % If normalization needed then do it
        if iNorm ~= 3
          DataMatrix2 = zscore(DataMatrix,[],iNorm);
        else
          DataMatrix2 = DataMatrix;
        end


        % Performing PCA
        [~, Score, Eigen] = princomp(DataMatrix2,'econ');
        
        % Calculating cumulative variance
        CumEigen = cumsum(Eigen)/sum(Eigen);
        
        % Amount of variance explained by dimensionality reduction
        for iVar = 1:length(Var)

          % Choosing the number of dimensions by cumulative variance
          MaxId = find(CumEigen>Var(iVar),1);

          % Data after the PCA transformation
          DataMatrix3 = Score(:,1:MaxId);

          for iC = 1:length(C)
            
            % Catching errors if linear SVM does not converge
            try
              
              % Init
              ErrorLinear = zeros(1,5);
              
              % Different personality traits
              for ii = 1:5
                
                % linear SVM, function handle
                ClassFun = @(xtrain,ytrain,xtest)(svmclassify(...
                  svmtrain(xtrain,ytrain,'boxconstraint',C(iC),...
                  'kktviolationlevel',0.01,...
                  'options',optimset('maxiter',1000000)),xtest));
                
                % Performing the classification
                ErrorLinear(ii) = crossval('mcr',DataMatrix3,...
                  Classes(:,ii),'predfun',ClassFun,'partition',CVpartitions);
              end
              
              % Saving results
              Results(end+1,:) = [ErrorLinear, mean(ErrorLinear), ...
                iDataType, iRoi, iFreq, iNorm, iVar, iC, 1, ...
                NaN, NaN, NaN];
              
            catch
              
              % Saving NaNs if linear model did not converge
              Results(end+1,:) = [NaN, NaN, NaN, NaN, NaN, NaN, ...
                iDataType, iRoi, iFreq, iNorm, iVar, iC, 1, ...
                NaN, NaN, NaN];
            end

            % Error in RBF sigma estimation
            for iRBFerror = 1:length(RBFerror)

              % Finding RBF parameter for SVM with random classes

              ClassesRandInit = zeros(NUM_OF_SAMPLES,1);
              ClassesRandInit(1:ceil(NUM_OF_SAMPLES / 2)) = 1;
              ClassesRand = ClassesRandInit(randperm(NUM_OF_SAMPLES));
              
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
                
                % Save the results
                ErrorTemp = sum(abs(RE - ClassesRand)) / NUM_OF_SAMPLES;
                
                % If needed error is reached, break and continue
                if  ErrorTemp > RBFerror(iRBFerror)
                  RBFsigma = Values(jj);
                  break
                end
              end

              % Results for SVM with RBF kernel
              ErrorRBF = zeros(1,5);
              
              % Different personalities
              for ii = 1:5
                
                % RBF kernel SVM
                ClassFun = @(xtrain,ytrain,xtest)(svmclassify(svmtrain(...
                  xtrain,ytrain,'kernel_function','rbf',...
                  'rbf_sigma',RBFsigma,'boxconstraint',C(iC),...
                  'options',optimset('maxiter',100000)),xtest));
                ErrorRBF(ii) = crossval('mcr',DataMatrix3,...
                  Classes(:,ii),'predfun',ClassFun,'partition',CVpartitions);
              end
              
              % Saving results
              Results(end+1,:) = [ErrorRBF, mean(ErrorRBF), ...
                iDataType, iRoi, iFreq, iNorm, iVar, iC, 2, ...
                iRBFerror, RBFsigma, MaxId];
            end
            
            % Print out the progress
            fprintf(1,'Progress: %6.2f %%\n',size(Results,1)/ParemeterSpace*100);
          end
        end
      end
    end
  end
end

save('Results.mat','Results');
save('Partitions.mat','CVpartitions');
