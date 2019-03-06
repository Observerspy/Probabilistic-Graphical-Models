% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[P1 logll1 ClassProb1 PairProb1] = EM_HMM(datasetTrain(1).actionData, datasetTrain(1).poseData, G, datasetTrain(1).InitialClassProb, datasetTrain(1).InitialPairProb, maxIter);
[P2 logll2 ClassProb2 PairProb2] = EM_HMM(datasetTrain(2).actionData, datasetTrain(2).poseData, G, datasetTrain(2).InitialClassProb, datasetTrain(2).InitialPairProb, maxIter);
[P3 logll3 ClassProb3 PairProb3] = EM_HMM(datasetTrain(3).actionData, datasetTrain(3).poseData, G, datasetTrain(3).InitialClassProb, datasetTrain(3).InitialPairProb, maxIter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Classify each of the instances in datasetTrain
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

accuracy = 0;
predicted_labels = [];
N = size(datasetTest.poseData,1);
K = 3;
poseData = datasetTest.poseData;
P = [P1,P2,P3];
I = size(datasetTest.actionData, 2);
loglikelihood = zeros(I,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iclass = 1:3
  logEmissionProb = zeros(N,K);
  for i = 1:N
    for k=1:K
      for part=1:10
        parentpart = 0;
        if (G(part,1) == 1)
          parentpart = G(part, 2);
        end
        
        if ( parentpart == 0 )
          logEmissionProb(i,k) = logEmissionProb(i,k) + lognormpdf( poseData(i, part, 1), P(iclass).clg(part).mu_y(k), P(iclass).clg(part).sigma_y(k) );
          logEmissionProb(i,k) = logEmissionProb(i,k) + lognormpdf( poseData(i, part, 2), P(iclass).clg(part).mu_x(k), P(iclass).clg(part).sigma_x(k) );
          logEmissionProb(i,k) = logEmissionProb(i,k) + lognormpdf( poseData(i, part, 3), P(iclass).clg(part).mu_angle(k), P(iclass).clg(part).sigma_angle(k) );
        else
          mu_y = P(iclass).clg(part).theta(k, 1) + P(iclass).clg(part).theta(k, 2) * poseData(i, parentpart, 1) + P(iclass).clg(part).theta(k, 3) * poseData(i, parentpart, 2) + P(iclass).clg(part).theta(k, 4) * poseData(i, parentpart, 3);
          mu_x = P(iclass).clg(part).theta(k, 5) + P(iclass).clg(part).theta(k, 6) * poseData(i, parentpart, 1) + P(iclass).clg(part).theta(k, 7) * poseData(i, parentpart, 2) + P(iclass).clg(part).theta(k, 8) * poseData(i, parentpart, 3);
          mu_angle = P(iclass).clg(part).theta(k, 9) + P(iclass).clg(part).theta(k, 10) * poseData(i, parentpart, 1) + P(iclass).clg(part).theta(k, 11) * poseData(i, parentpart, 2) + P(iclass).clg(part).theta(k, 12) * poseData(i, parentpart, 3);
          logEmissionProb(i,k) = logEmissionProb(i,k) + lognormpdf( poseData(i, part, 1), mu_y, P(iclass).clg(part).sigma_y(k) );
          logEmissionProb(i,k) = logEmissionProb(i,k) + lognormpdf( poseData(i, part, 2), mu_x, P(iclass).clg(part).sigma_x(k) );
          logEmissionProb(i,k) = logEmissionProb(i,k) + lognormpdf( poseData(i, part, 3), mu_angle, P(iclass).clg(part).sigma_angle(k) );
        end
        
      endfor
    end
  end
  
  for i = 1:I
	  % construct all the three types of factors for each action and do inference to fill ClassProb and PairProb
	  m = length(datasetTest.actionData(i).marg_ind); % 1 to m represents S variables
	  F = repmat(struct('var',[],'card',[],'var',[]),1,2*m); % 1 is P(S1), 2 to m is P(Si/Si-1) and m+1 to 2m P(S/O)
	  F(1).var = 1; F(1).card = [K];F(1).val = log(P(iclass).c);
	  temp = log(reshape(P(iclass).transMatrix',1,9));
	  for j = 2:m
		  F(j).var = [j j-1]; F(j).card = [K K]; F(j).val = temp;
	  end
	  for j = 1:m
		  F(j+m).var = [j];F(j+m).card = [K];
		  F(j+m).val = logEmissionProb(datasetTest.actionData(i).marg_ind(j),:);
	  end
	  [M, PCalibrated] = ComputeExactMarginalsHMM(F);
	  loglikelihood(i,iclass) += logsumexp(PCalibrated.cliqueList(1).val);
  end
end
[temp, predicted_labels] = max(loglikelihood,[],2);
if isfield(datasetTest, 'labels')
  accuracy = sum(predicted_labels==datasetTest.labels)/I;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
