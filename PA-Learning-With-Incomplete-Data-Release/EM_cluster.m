% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  P.c=mean(ClassProb,1);
  P.clg = repmat(struct('mu_y',[],'sigma_y',[],'mu_x',[],'sigma_x',[],'mu_angle',[],'sigma_angle',[],'theta',[]), 1, size(poseData,2));
  
  for part = 1:10
    for k=1:K
      parentpart = 0;
      if (G(part, 1) == 1)
        parentpart = G(part, 2);
      end

      if parentpart == 0
        [mu, sigma] = FitG(poseData(:, part, 1), ClassProb(:, k));
        P.clg(part).mu_y(k) = mu;
        P.clg(part).sigma_y(k) = sigma;

        [mu, sigma] = FitG(poseData(:, part, 2), ClassProb(:, k));
        P.clg(part).mu_x(k) = mu;
        P.clg(part).sigma_x(k) = sigma;

        [mu, sigma] = FitG(poseData(:, part, 3), ClassProb(:, k));
        P.clg(part).mu_angle(k) = mu;
        P.clg(part).sigma_angle(k) = sigma;

      else
        U = [];
        U(:, 1) = poseData(:, parentpart, 1);
        U(:, 2) = poseData(:, parentpart, 2);
        U(:, 3) = poseData(:, parentpart, 3);

        [Beta, sigma] = FitLG(poseData(:, part, 1), U, ClassProb(:, k));
        P.clg(part).theta(k, 1) = Beta(4);
        P.clg(part).theta(k, 2:4) = Beta(1:3);
        P.clg(part).sigma_y(k) = sigma;

        [Beta, sigma] = FitLG(poseData(:, part, 2), U, ClassProb(:, k));
        P.clg(part).theta(k, 5) = Beta(4);
        P.clg(part).theta(k, 6:8) = Beta(1:3);
        P.clg(part).sigma_x(k) = sigma;

        [Beta, sigma] = FitLG(poseData(:, part, 3), U, ClassProb(:, k));
        P.clg(part).theta(k, 9) = Beta(4);
        P.clg(part).theta(k, 10:12) = Beta(1:3);
        P.clg(part).sigma_angle(k) = sigma;

      end
    end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i=1:N
    for k=1:K
      ClassProb(i,k) = ClassProb(i,k) + log(P.c(k));
      for part=1:10
        parentpart = 0;
        if (G(part,1) == 1)
          parentpart = G(part, 2);
        end
        
        if ( parentpart == 0 )
          ClassProb(i,k) = ClassProb(i,k) + lognormpdf( poseData(i, part, 1), P.clg(part).mu_y(k), P.clg(part).sigma_y(k) );
          ClassProb(i,k) = ClassProb(i,k) + lognormpdf( poseData(i, part, 2), P.clg(part).mu_x(k), P.clg(part).sigma_x(k) );
          ClassProb(i,k) = ClassProb(i,k) + lognormpdf( poseData(i, part, 3), P.clg(part).mu_angle(k), P.clg(part).sigma_angle(k) );
        else
          mu_y = P.clg(part).theta(k, 1) + P.clg(part).theta(k, 2) * poseData(i, parentpart, 1) + P.clg(part).theta(k, 3) * poseData(i, parentpart, 2) + P.clg(part).theta(k, 4) * poseData(i, parentpart, 3);
          mu_x = P.clg(part).theta(k, 5) + P.clg(part).theta(k, 6) * poseData(i, parentpart, 1) + P.clg(part).theta(k, 7) * poseData(i, parentpart, 2) + P.clg(part).theta(k, 8) * poseData(i, parentpart, 3);
          mu_angle = P.clg(part).theta(k, 9) + P.clg(part).theta(k, 10) * poseData(i, parentpart, 1) + P.clg(part).theta(k, 11) * poseData(i, parentpart, 2) + P.clg(part).theta(k, 12) * poseData(i, parentpart, 3);
          ClassProb(i,k) = ClassProb(i,k) + lognormpdf( poseData(i, part, 1), mu_y, P.clg(part).sigma_y(k) );
          ClassProb(i,k) = ClassProb(i,k) + lognormpdf( poseData(i, part, 2), mu_x, P.clg(part).sigma_x(k) );
          ClassProb(i,k) = ClassProb(i,k) + lognormpdf( poseData(i, part, 3), mu_angle, P.clg(part).sigma_angle(k) );
        end
        
      endfor
    endfor
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  loglikelihood(iter) = 0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  lse = logsumexp(ClassProb);
  loglikelihood(iter) = sum(lse);
  ClassProb = exp(ClassProb-repmat(lse,1,K));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
