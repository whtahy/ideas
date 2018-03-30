## Auto differentiation
- Computational graph + chain rule


## Feed forward architecture
- X -> [Linear -> Nonlinear] -> Y


## Activation functions
- ReLU
- SELU (Self-Normalizing NN)
- Arcsinh
- Softmax/logistic


## Regularization
- L2
- Inverted dropout
- Data augmentation
- Early stopping (maybe)


## Train/dev/test
- Keep dev + test on target! (drawn from same distribution!)
- Change dev -> change test!
- More data -> smaller dev/test: 60/20/20 -> 98/1/1
- Test: May not be necessary (test <-> dev)
- NN robust to random errors in train data
- NN less robust to systematic errors
- Training-dev set: train and dev/test have different distributions
- Train -> train-dev: variance
- Train-dev -> dev: bias


## Data Preprocessing
- Zero mean, unit variance
- Restricted range (eg, 0 to 1 for pixel data)


## Weight initialization
- Zero mean, unit variance


## Optimization
- Gradient descent: every epoch uses every training example
- Batch size: 64 to 512, 1024, 2056
- Adam: RMSprop + momentum
- Learning rate decay: smooth, step, manual
- Local optima: unlikely to get stuck, but saddle points can slow learning


## Hyperparameter Tuning
- Priority:
    1.  alpha
    2.  \# nodes, momentum beta, batch size
    3.  \# layers, learning rate decay
- Random search (not grid search)
- Coarse to fine
- Linear scale: # nodes, # layers
- Log scale: learning rate, momentum beta


## Orthogonalization
- 1 knob -> 1 effect
- Train set (bias): bigger network
- Dev set (var): regularization, bigger train set
- Test set: bigger dev set
- Real world: change dev set or cost function
- Early stopping violates this rule! (-> train set + dev set)


## Metrics
- Optimizing metric: unbounded, primary
- Satisficing metric: threshold (hard/soft), secondary
- If dev/test perf is bad, change dev/test and/or metric!


## Human level performance
- Help measure bias + variance
- Proxy for Bayes optimal error: expert performance (not average)
- Avoidable bias: delta between model & Bayes error (unavoidable bias)
- Below HLP: manual error analysis, more interpretable bias/var metrics
- Above HLP: hard to measure/distinguish bias vs variance


## Error analysis
- Examine dev set error, incl mislabeled data
- Ceiling analysis: compare options vs potential upside
- Mislabeled data: beware when comparing models
- Also examine correct predictions for mislabel error

