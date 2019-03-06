function [Assignments, Cost] = FuzzyKMean(dataset, K, maxIter, window)
  N = size(dataset, 1);

  # normalize
  numFeatures = prod(size(dataset)(2:end));
  dataset = reshape(dataset, N, numFeatures);
  dataset .-= repmat(mean(dataset), N, 1);
  dataset ./= repmat(std(dataset), N, 1);

  Centroids = zeros(K, numFeatures);
  Similarities = zeros(N, K);
  Assignments = rand(N, K);
  Assignments ./= repmat(sum(Assignments, 2), 1, K);
  Assignments = log(Assignments);

  for iter = 1:maxIter
    for k = 1:K
      memberships = 2 .* Assignments(:, k);

      Centroids(k, :) = exp(logsumexp(log(dataset') .+ repmat(memberships, 1, numFeatures)') .- ...
                            logsumexp(memberships'));
    end

    for i = 1:N
      for k = 1:K
        a = dataset(i, :);
        c = Centroids(k, :);
        Similarities(i, k) = - norm(a - c) .^ 2;
      end
    end

    Assignments = Similarities .- repmat(logsumexp(Similarities), 1, K);

    Cost(iter) = logsumexp((Similarities .+ Assignments)(:)');
    if iter > window && Cost(iter) == Cost(iter - window)
      break
    end
  end

  Assignments = exp(Assignments);
end
