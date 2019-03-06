% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
OptimalDecisionRule = CalculateExpectedUtilityFactor(I);
 
 
 
if length(OptimalDecisionRule.var) == 1
 
    [MEU, index] = max(OptimalDecisionRule.val);
 
    OptimalDecisionRule.val = zeros(size(OptimalDecisionRule.val));
 
    OptimalDecisionRule.val(index) = 1;
 
else
 
    assignments = IndexToAssignment(1 : prod(OptimalDecisionRule.card(1 : end - 1)), OptimalDecisionRule.card(1 : end - 1));
 
    MEU = 0;
 
    for ii = 1 : OptimalDecisionRule.card(end)
 
        indices1 = AssignmentToIndex([assignments, ii * ones(size(assignments, 1), 1)], OptimalDecisionRule.card);
 
        [meu, indices2] = max(OptimalDecisionRule.val(indices1));
 
        MEU = MEU + meu;
 
        OptimalDecisionRule.val(indices1) = 0;
 
        OptimalDecisionRule.val(indices1(indices2)) = 1;
 
    end
 
end
end
