%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isMax
  for i = 1:N
    P.cliqueList(i).val = log(P.cliqueList(i).val);
  end
endif
[i j]=GetNextCliques(P, MESSAGES);
while i && j
  factor=P.cliqueList(i);
  for k=1:N
    if k ~= j
      if isMax
        factor = FactorSum(factor, MESSAGES(k, i));
      else
        factor = FactorProduct(factor, MESSAGES(k,i));
      end
    endif
  endfor
  
  sepset = intersect(P.cliqueList(i).var, P.cliqueList(j).var);
  V=setdiff(factor.var,sepset);
  if isMax
    MESSAGES(i,j)=FactorMaxMarginalization(factor, V);
  else
    MESSAGES(i,j)=FactorMarginalization(factor, V);
    MESSAGES(i,j).val=MESSAGES(i,j).val./sum(MESSAGES(i,j).val);
  endif

  [i j]=GetNextCliques(P, MESSAGES);
endwhile

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N
  for k = 1:N
    if P.edges(i,k)
      if isMax
        P.cliqueList(i) = FactorSum(P.cliqueList(i), MESSAGES(k,i));
      else
        P.cliqueList(i) = FactorProduct(P.cliqueList(i), MESSAGES(k,i));
      end
    end
  end
end


return
