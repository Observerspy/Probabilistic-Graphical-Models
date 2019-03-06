% Copyright (C) Daphne Koller, Stanford University, 2012

function EU = SimpleCalcExpectedUtility(I)

  % Inputs: An influence diagram, I (as described in the writeup).
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return Value: the expected utility of I
  % Given a fully instantiated influence diagram with a single utility node and decision node,
  % calculate and return the expected utility.  Note - assumes that the decision rule for the 
  % decision node is fully assigned.

  % In this function, we assume there is only one utility node.
  F = [I.RandomFactors I.DecisionFactors];
  U = I.UtilityFactors(1);
  EU = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  V = unique([F(:).var]);
  P = setdiff(V, U.var); %not U parent
  F_new = VariableElimination(F, P);
  F_product = F_new(1);
  for i = 2:length(F_new)
    F_product = FactorProduct(F_product, F_new(i));
  end
  F_product = FactorProduct(F_product,U);
  EU = sum(F_product.val);
end
