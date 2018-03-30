## Architecture
- Auto differentiation: computational graph + chain rule -> backprop
- Feed forward: X -> [Linear -> Nonlinear] -> Y


## Activation Functions
- ReLU
- SELU (Self-Normalizing NN)
- Arcsinh
- Softmax/logistic (mutually exclusive)
- Multilabel logistic (overlapping)


## Regularization
- L2
- Inverted dropout
- Data augmentation
- Early stopping (maybe)


## Train/Dev/Test
- Keep dev + test on target!  Same distribution: change dev -> change test!
- More data -> smaller dev/test: 60/20/20 -> 98/1/1
- Test: May not be necessary (test <-> dev)
- NN robust to *random* error in train data, less robust to *systematic* error
- Training-dev set: when train and dev/test have different distributions


## Data Preprocessing
- Zero mean, unit variance
- Restricted range (eg, 0 to 1 for pixel data)


## Weight Initialization
- Zero mean, unit variance


## Optimization
- Gradient descent: every epoch uses every training example
- Batch size: 64 to 512, 1024, 2056, etc
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


## Human Level Performance
- Help measure bias + variance
- Proxy for Bayes optimal error: expert performance (not average)
- Avoidable bias: delta between model & Bayes error (unavoidable bias)
- Below HLP: manual error analysis, more interpretable bias/var metrics
- Above HLP: hard to measure/distinguish bias vs variance


## Error Analysis
- Examine dev set error, incl mislabeled data
- Eg, 100 examples by hand, in spreadsheet
- Ceiling analysis: compare options vs potential upside
- Mislabeled data: beware when comparing models
- Also examine correct predictions for mislabel error
- Data mismatch: identify the mismatch, augment train
- Data synthesis: eg, speech + car noise -> in car speech (beware overfitting!)


## Types of Error
- Bayes/HLP -> Train: bias
- Train -> train-dev: variance
- Train-dev -> dev: data mismatch
- Dev -> Test: overfit dev

<center>
|           | Train Distribution |                   | Dev/Test Distribution |
|:---------:|:------------------:|:-----------------:|:---------------------:|
| HLP/Bayes | Human level        |                   |                       |
|           | **Avoidable bias** |                   |                       |
| Train     | Train error        |                   |                       |
|           | **Variance**       |                   |                       |
| Not Train | Train-dev error    | **Data mismatch** | Dev/Test error        |
</center>


## Transfer Learning
- Replace output layer with 1 or more new layers
- Pre-training: train just new layers
- Fine-tuning: then train full net (if you have lots of data)
- Used to transfer from large data set -> small data set, with shared features


## Multi-Task Learning
- Multidimensional target: eg, y = [pedestrian, car, stop sign, traffic light]
- Use N targets instead of N networks: eg, multilabel classification
- Use when: shared features, each task has similar amount of data
- Can use sparse/missing data: omit missing data from loss function
- Need a bigger network vs smaller dim target


## Pipeline
- End to end: requires large data sets: machine translation
- Multistage: eg, bounding box -> face recognition -> general purpose ID system

