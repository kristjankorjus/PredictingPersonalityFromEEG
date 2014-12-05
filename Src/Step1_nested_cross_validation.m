function [ Parameters ] = Step1_nested_cross_validation( DataOpen, DataClose, ...
  Classes, k_fold, partition_ID)
% Step1_nested_cross_validation does a cross-validation with all the models

% Function performs a nested cross-validation step for the article 
% "Personality cannot be predicted from the power of resting state 1 EEG"
% submitted to Frontiers in Human Neuroscience
%
% Output: the best hyper-parameters for each personality trait
%
% Tested with Windows 8 and Matlab 2013a with Statistics Toolbox
%
% Comments and questions: Kristjan Korjus (korjus@gmail.com)

%% Finding best hyper parameters with k-fold cross-validation

% Fixed parameters
num_of_samples = size(DataOpen,1);
num_of_classes = size(Classes,2);

%%%%%%%%%%%%%%%%%%%
% Parameter space %
%%%%%%%%%%%%%%%%%%%

% Using different types of data:
DataType = 1:2; % Eyes open, eyes close

% Different Regions of Interests (ROI):
Roi = 1:2; % All, Literature

% Different frequency bands:
Freq = 1:3; % All, customized pooling, based on literature

% Normalization:
Norm = 1:3; % Column normalization, Row normalization, No normalization

% Choosing number of dimensions by total variance explained by PCA:
Var = [0.7, 0.9, NaN]; % Variance explained by PCA or no PCA

% C parameters for SVM
C = [0.01, 100];

% Choosing the model:
% Model = 1:2; % Linear SVM, SVM with RBF kernel

% Finding RBF sigma parameter for SVM:
RBFerror = [0.1, 0.3]; % RBF sigma parameter error rate for non-linear SVM

ParemeterSpace = length(DataType)*length(Roi)*length(Freq)*length(Norm)*...
  length(Var)*length(C)*3;

% Random Partitions
partitions = crossvalind('Kfold', num_of_samples, k_fold);

%% Master for-loop

% Results
Results=[];

% Different data used
for iDataType = DataType
  
  if iDataType == 1
    Data = DataOpen;
  elseif iDataType == 2
    Data = DataClose;
  else
    Data = [DataOpen,DataClose];
    clear('DataOpen');
    clear('DataClose');
  end
  
  % Size of the data
  [num_of_samples, NumOfChannels, NumOfSpectrumPoints] = size(Data);
  
  % Different Regions of Interests
  for iRoi = Roi

    % If must take ROIs
    if iRoi == 2
      
      %Temporary variable for single data
      DataTemp = zeros(num_of_samples,7,NumOfSpectrumPoints);
      
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
        DataTemp=zeros(num_of_samples,7,NumOfSpectrumPoints);
    
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
        DataMatrix = zeros(num_of_samples, NumOfChannels*NumOfSpectrumPoints);
        
        % For all samples
        for ii = 1:num_of_samples
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
        DataMatrix = zeros(num_of_samples,NumOfChannels*65);

        for ii = 1:num_of_samples
          count = 0;
          for kk = 1:NumOfChannels;
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
        DataMatrix = zeros(num_of_samples,NumOfChannels*10);

        for ii = 1:num_of_samples
          for jj = 1:NumOfChannels
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

          if ~isnan(Var(iVar))
            % Choosing the number of dimensions by cumulative variance
            MaxId = find(CumEigen>Var(iVar),1);

            % Data after the PCA transformation
            DataMatrix3 = Score(:,1:MaxId);
          else
            DataMatrix3 = DataMatrix2;
          end

   
          for iC = 1:length(C)
            
            % Catching errors if linear SVM does not converge
            try
            % Nested-Cross-Validation
            
              % Init
              CorrectLinear = zeros(1,num_of_classes);
              
              for i_partition = 1:k_fold

                % Indeces for test and train
                test_id = (partitions == i_partition);
                train_id = ~test_id;

                % Getting the data
                xtrain = DataMatrix3(train_id,:);
                ytrain = Classes(train_id,:);

                xtest = DataMatrix3(test_id,:);
                ytest = Classes(test_id,:);

                % Different personality traits
                for ii = 1:num_of_classes

                  % linear SVM, function handle
                  classified_classes = svmclassify(...
                    svmtrain(xtrain,ytrain(:,ii),'boxconstraint',C(iC),...
                    'kktviolationlevel',0.01,...
                    'options',optimset('maxiter',1000000)),xtest);

                  % Adding if the result is correct
                  CorrectLinear(ii) = CorrectLinear(ii) + ...
                    sum(classified_classes == ytest(:,ii));
                end
              end
                
              ErrorLinear = (num_of_samples - CorrectLinear)/num_of_samples;
              
              % Saving results
              Results(end+1,:) = [ErrorLinear, mean(ErrorLinear), ...
                iDataType, iRoi, iFreq, iNorm, iVar, iC, 1, ...
                NaN, NaN, MaxId];

            catch

              % Saving NaNs if linear model did not converge
              Results(end+1,:) = [nan(1,num_of_classes+1), ...
                iDataType, iRoi, iFreq, iNorm, iVar, iC, 1, ...
                NaN, NaN, MaxId];
              
            end

            % Error in RBF sigma estimation
            for iRBFerror = 1:length(RBFerror)

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
                
                % Save the results
                ErrorTemp = sum(abs(RE - ClassesRand)) / num_of_samples;
                
                % If needed error is reached, break and continue
                if  ErrorTemp > RBFerror(iRBFerror)
                  RBFsigma = Values(jj);
                  break
                end
              end
          
              % Results for SVM with RBF kernel
              CorrectRBF = zeros(1,num_of_classes);
                            
              % Nested-Cross-Validation
              for i_partition = 1:k_fold
                
                % Indeces for test and train
                test_id = (partitions == i_partition);
                train_id = ~test_id;
                
                % Getting the data
                xtrain = DataMatrix3(train_id,:);
                ytrain = Classes(train_id,:);

                xtest = DataMatrix3(test_id,:);
                ytest = Classes(test_id,:);

                % Different personalities
                for ii = 1:num_of_classes
                  % RBF kernel SVM
                  classified_classes = svmclassify(svmtrain(...
                    xtrain,ytrain(:,ii),'kernel_function','rbf',...
                    'rbf_sigma',RBFsigma,'boxconstraint',C(iC),...
                    'options',optimset('maxiter',100000)),xtest);
                  
                  % Adding if the result is correct
                  CorrectRBF(ii) = CorrectRBF(ii) + ...
                    sum(classified_classes == ytest(:,ii));
                end
              end
              
              ErrorRBF = (num_of_samples - CorrectRBF) / num_of_samples;
              
              % Saving results
              Results(end+1,:) = [ErrorRBF, mean(ErrorRBF), ...
                iDataType, iRoi, iFreq, iNorm, iVar, iC, 2, ...
                iRBFerror, RBFsigma, MaxId];
            end
            
            % Print out the progress
            fprintf(1,'Inside partition %6.2f, nested CV: %6.2f %%\n',...
              partition_ID, size(Results,1)/ParemeterSpace*100);
            
          end
        end
      end
    end
  end
end

% Choosing and saving only the best parameters
Parameters = zeros(num_of_classes,size(Results,2));
for ii = 1:num_of_classes
  row = find(Results(:,ii) == min(Results(:,ii)),1);
  Parameters(ii,:) = Results(row,:);
end

% Saving the results just in case
save(strcat('Results',num2str(partition_ID),'.mat'),'Results');

end
