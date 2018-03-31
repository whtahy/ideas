# Basics


## Architecture
- Backprop: computational graph + chain rule -> auto differentiation
- (FNN) Feedforward: `X -> [Linear -> Nonlinear] -> Y`
- (CNN) Convolution: dot product filter/kernel (aka "Cross-Correlation")


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
- Keep `dev` + `test` on target!  Same distribution: change dev -> change test!
- More data -> smaller dev/test: `60/20/20` -> `98/1/1`
- Test: May not be necessary (`test` <-> `dev`)
- NN robust to *random* error on `train`, less robust to *systematic* error
- `training-dev`: train and dev/test have different distributions


## Data Preprocessing
- Zero mean, unit variance
- Restricted range (eg, `0` to `1` for pixel data)


## Weight Initialization
- Zero mean, unit variance


## Optimization
- Gradient descent: every epoch uses every training example
- Batch size: `64` to `512`, `1024`, `2056`, etc
- Learning rate decay: smooth, step, manual
- Local optima: unlikely to get stuck, but saddle points can slow learning
- Adam: RMSprop + momentum, robust convergence
- (Batch) SGD + Momentum: best performance, but requires fine tuning


## Hyperparameter Tuning
- Priority:
    1.  `alpha`
    2.  \# nodes, momentum `beta`, batch size
    3.  \# layers, learning rate decay
- Random search (not grid search)
- Coarse to fine
- Linear scale: \# nodes, \# layers
- Log scale: learning rate, momentum beta


## Orthogonalization
- 1 knob -> 1 effect

| Source         | Try                            |
| ---            | ---                            |
| `train` (bias) | Bigger network                 |
| `dev` (var)    | Regularization, bigger `train` |
| `test`         | Bigger `dev`                   |
| Real world     | Change `dev` or cost function  |

- Early stopping violates this rule! (-> `train` *and* `dev`)


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
- Examine `dev` error, incl mislabeled data
- Eg, 100 examples by hand, in spreadsheet
- Ceiling analysis: compare options vs potential upside
- Mislabeled data: beware when comparing models
- Also examine correct predictions for mislabel error
- Data mismatch: identify the mismatch, augment train
- Data synthesis: eg, speech + car noise -> in car speech (beware overfitting!)


## Types of Error
- Bayes/HLP -> `train`: bias
- `train` -> `train-dev`: variance
- `train-dev` -> `dev`: data mismatch
- `dev` -> `test`: overfit dev

|           | Train Distribution |                   | Dev/Test Distribution |
| :---:     | :---:              | :---:             | :---:                 |
| HLP/Bayes | "Human level perf" |                   | ?                     |
|           | **Avoidable bias** |                   | **Avoidable bias**    |
| Seen      | Train error        |                   | ?                     |
|           | **Variance**       |                   |                       |
| Unseen    | Train-dev error    | **Data mismatch** | Dev/Test error        |


## Transfer Learning
- Replace output layer with 1+ new layers
- Pre-training: train just new layers
- Fine-tuning: then train full net (if you have lots of data)
- Transfer large -> small data set, with shared features


## Multi-Task Learning
- Multidimensional target: eg, `y = [pedestrian, car, stop sign, traffic light]`
- Use `N` targets instead of `N` networks: eg, multilabel classification
- Use when: shared features, each task has similar amount of data
- Sparse/missing data: just omit missing data from loss function
- Need a bigger network vs individual / smaller dim target(s)


## Pipeline
- End to end: requires large data sets: machine translation
- Multistage: eg, bounding box -> face recognition -> general purpose ID system




# Convolutional Neural Networks


## Examples
- GoogLeNet, aka Inception-v1 (2014 ILSVRC \#1)
- VGGNet (2014 ILSVRC \#2)
- DenseNet
- ResNet (2015 ILSVRC \#1)


## Assumptions

> Secondly, a deficiency of fully-connected architectures is that the topology of the input is entirely ignored.  The input variables can be presented in any (fixed) order without affecting the outcome of the training.  On the contrary, images (or time-frequency representations of speech) have a strong 2D local structure: variables (or pixels) that are spatially or temporally nearby are highly correlated.  Local correlations are the reasons for the well-known advantages of extracting and combining *local* features before recognizing spatial and temporal objects, because configurations of neighboring variables can be classified into a small number of categories (eg, edges, corners, etc). *Convolutional Networks* force the extraction of local features by restricting the receptive fields of hidden units to be local.

--LeCun et al (1998), *Gradient-Based Learning Applied to Document Recognition*.

- Parameter sharing: features useful in 1 part of image are useful in other(s)
- Sparsity of connections (less param than FC): small local parameter sharing
- Features shared globally: small filters applied across whole image
- Translation invariance
- Eg, bad fit for centered/normalized face detection
    - Non global features / segmentation: nose features in nose region only, etc


## Layers
- Convolution (`conv`): dot product + bias -> nonlinear (eg, ReLU)
- Pooling (`pool`): downsample (no learning), eg max with `f = s = 2`, `p = 0'
- Fully connected ('fc'): FNN layer


## Filter/Kernel
- Learn weights (ie, matrix elements) via backprop
- Dimensions: `height x width x channels/depth (3D)`
- Dimension shrinkage: eg, `3D (RGB) * 3D (filter) -> 2D x (# filters)`
- Stride (s): step size, round down
- \# parameters invariant to input size: `h x w x ch x (# f)`
- Size: `3x3`, `5x5`, `7x7`, odd `f` -> symmetric padding + central pixel (loc)


## Padding (p)
- Avoid shrinkage
- Use data more uniformly: corner vs center pixel
- Valid: no padding (`p = 0`)
- Same: no shrinkage (eg, `p = (f-1)/2` for `s = 1`)



















