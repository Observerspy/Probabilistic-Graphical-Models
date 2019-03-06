% You should put all your code for recognizing unknown actions in this file.
% Describe the method you used in YourMethod.txt.
% Don't forget to call SavePrediction() at the end with your predicted labels to save them for submission, then submit using submit.m

function [accuracy, predicted_labels] = RecognizeUnknownActions(datasetTrain, datasetTest, G, maxIter)
  for i = 1:length(datasetTrain)
    Assignments = FuzzyKMean(datasetTrain(i).poseData, 3, 50, 5);
    datasetTrain(i).InitialClassProb = Assignments;

    ## datasetTrain(i).InitialPairProb = zeros(size(datasetTrain(i).InitialPairProb, 1), K * K);
    # Init pair class prob from sufficient statistics
    ## for l = 1:length(datasetTrain(i).actionData)
    ##   for p = 1:length(datasetTrain(i).actionData(l).pair_ind)
    ##     from = datasetTrain(i).InitialClassProb(datasetTrain(i).actionData(l).marg_ind(p), :);
    ##     to = datasetTrain(i).InitialClassProb(datasetTrain(i).actionData(l).marg_ind(p + 1), :);
    ##     datasetTrain(i).InitialPairProb(datasetTrain(i).actionData(l).pair_ind(p), :) = (from' * to)(:);
    ##   end
    ## end
  end

  [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter);
  SavePredictions(predicted_labels);
end