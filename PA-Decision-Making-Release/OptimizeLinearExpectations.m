% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  MEU = [];
  OptimalDecisionRule = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  D = I.DecisionFactors(1);
  OptimalDecisionRule = struct('var', [], 'card', [], 'val', []);
  for i=1:length(I.UtilityFactors)
    Inew = I;
    Inew.UtilityFactors = I.UtilityFactors(i);
    OptimalDecisionRule = FactorSum(OptimalDecisionRule, CalculateExpectedUtilityFactor( Inew ));
  end 
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
