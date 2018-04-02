# Table of Contents

- 1 [Basics](#1-basics)
    - 1.1 [Architecture](#11-architecture)
    - 1.2 [Setup](#12-setup)
    - 1.3 [Training](#13-training)
    - 1.4 [Iteration](#14-iteration)

- 2 [CNN](#2-cnn-convolutional-neural-networks)
    - 2.1 [Image Recognition](#21-image-recognition)
    - 2.2 [Object Detection](#22-object-detection)
    - 2.3 [Extras](#23-extras)

- 3 [Sequence Models](#3-sequence-models)


--------------------------------------------------------------------------------


# 1 Basics


## 1.1 Architecture

### Tricks
- Backprop: computational graph + chain rule -> auto differentiation
- Residual block / skip connection
- Inception module
- (FNN) Feedforward: `X -> [Linear -> Nonlinear] -> Y`
- (CNN) Convolution: dot product filter/kernel (aka "Cross-Correlation")

### Activation Functions
- ReLU
- SELU (Self-Normalizing NN)
- Arcsinh
- Softmax/logistic (mutually exclusive)
- Multilabel logistic (overlapping)

### Regularization
- L2
- Inverted dropout
- Data augmentation
- Early stopping (maybe)

### Transfer Learning
- Replace output layer with 1+ new layers
- Pre-training: train just new layers
- Fine-tuning: then train full net (if you have lots of data)
- Transfer large -> small data set, with shared features

### Multi-Task Learning
- Multidimensional target: eg, `y = [pedestrian, car, stop sign, traffic light]`
- Use `N` targets instead of `N` networks: eg, multilabel classification
- Use when: shared features, each task has similar amount of data
- Sparse/missing data: just omit missing data from loss function
- Need a bigger network vs individual / smaller dim target(s)

### Pipeline
- End to end: requires large data sets: machine translation
- Multistage: eg, bounding box -> face recognition -> general purpose ID system


## 1.2 Setup

### Train/Dev/Test
- Keep `dev` + `test` on target!  Same distribution: change dev -> change test!
- More data -> smaller dev/test: `60/20/20` -> `98/1/1`
- Test: May not be necessary (`test` <-> `dev`)
- NN robust to *random* error on `train`, less robust to *systematic* error
- `training-dev`: train and dev/test have different distributions

### Data Preprocessing
- Zero mean, unit variance
- Restricted range (eg, `0` to `1` for pixel data)

### Weight Initialization
- Zero mean, 0.01 stdev


## 1.3 Training

### Optimization
- Gradient descent: every epoch uses every training example
- Batch size: `64` to `512`, `1024`, `2056`, etc
- Learning rate decay: smooth, step, manual
- Local optima: unlikely to get stuck, but saddle points can slow learning
- Adam: RMSprop + momentum, robust convergence
- (Batch) SGD + Momentum: best performance, but requires fine tuning

### Hyperparameter Tuning
- Priority:
    1.  `alpha`
    2.  no. nodes, momentum `beta`, batch size
    3.  no. layers, learning rate decay
- Random search (not grid search)
- Coarse to fine
- Linear scale: no. nodes, no. layers
- Log scale: learning rate, momentum beta

### Orthogonalization
- 1 knob -> 1 effect

| Source         | Try                            |
| ---            | ---                            |
| `train` (bias) | Bigger network                 |
| `dev` (var)    | Regularization, bigger `train` |
| `test`         | Bigger `dev`                   |
| Real world     | Change `dev` or cost function  |

- Early stopping violates this rule! (-> `train` *and* `dev`)


## 1.4 Iteration

### Metrics
- Optimizing metric: unbounded, primary
- Satisficing metric: threshold (hard/soft), secondary
- If dev/test perf is bad, change dev/test and/or metric!

### Types of Error
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

### Human Level Performance
- Help measure bias + variance
- Proxy for Bayes optimal error: expert performance (not average)
- Avoidable bias: delta between model & Bayes error (unavoidable bias)
- Below HLP: manual error analysis, more interpretable bias/var metrics
- Above HLP: hard to measure/distinguish bias vs variance

### Error Analysis
- Examine `dev` error, incl mislabeled data
- Eg, 100 examples by hand, in spreadsheet
- Ceiling analysis: compare options vs potential upside
- Mislabeled data: beware when comparing models
- Also examine correct predictions for mislabel error
- Data mismatch: identify the mismatch, augment train
- Data synthesis: eg, speech + car noise -> in car speech (beware overfitting!)


--------------------------------------------------------------------------------


# 2 CNN: Convolutional Neural Networks


## 2.1 Image Recognition

### Examples
- GoogLeNet, aka Inception-v1 (2014 ILSVRC \#1)
    - Inception module: parallel/stacked layer
    - `1x1` convolution as bottleneck (reduce computation cost)
- ResNet (2015 ILSVRC \#1)
    - Shortcut/skip connection: preserves ability for layers to learn identity
- VGGNet (2014 ILSVRC \#2)
- DenseNet

### Assumptions
> Secondly, a deficiency of fully-connected architectures is that the topology of the input is entirely ignored.  The input variables can be presented in any (fixed) order without affecting the outcome of the training.  On the contrary, images (or time-frequency representations of speech) have a strong 2D local structure: variables (or pixels) that are spatially or temporally nearby are highly correlated.  Local correlations are the reasons for the well-known advantages of extracting and combining *local* features before recognizing spatial and temporal objects, because configurations of neighboring variables can be classified into a small number of categories (eg, edges, corners, etc). *Convolutional Networks* force the extraction of local features by restricting the receptive fields of hidden units to be local.

--LeCun et al (1998), *Gradient-Based Learning Applied to Document Recognition*

- Image recognition: 1 image -> 1 label
- Local: nearby features are correlated (sub-sampling)
- Global: same feature set across whole image, eg translation invariance
- Sparse: feature set is small/local -> fewer param than an equiv FC net
- Counterexample: centered/normalized face detection
    - Non global / segmented: nose features in nose region only, etc

### Layers
- Convolution (`conv`): dot product + bias -> nonlinear (eg, ReLU)
- Pooling (`pool`): downsample (no learning), eg max with `f = s = 2`, `p = 0`
- Fully connected (`fc`): FNN layer

### Filter/Kernel
- Learn weights (ie, matrix elements) via backprop
- Dimensions: `height x width x channels/depth (3D)`
- Dimension shrinkage: eg, `3D (RGB) * 3D (filter) -> 2D x (no. filters)`
- Stride (s): step size, round down
- no. parameters invariant to input size: `h x w x ch x (no. f)`
- Size: `3x3`, `5x5`, `7x7`, odd `f` -> symmetric padding + central pixel (loc)
- Network in network: `1x1` convolution -> influences no. channels

### Padding
- Avoid shrinkage
- Use data more uniformly: corner vs center pixel
- Valid: no padding (`p = 0`)
- Same: no shrinkage (eg, `p = (f-1)/2` for `s = 1`)

### Data Augmentation
- Computer vision is especially data hungry
- Vertical mirror
- Cropping
- Rotation, shearing, local warping
- Color shifting, PCA color augmentation (AlexNet)
- Multi-crop on `test`


## 2.2 Object Detection

### Main Ideas
- Bounding box & 1 image -> 1+ labels + bounding boxes
- Localization: 1 image -> 1 bounding box
- Detection: 1 image -> multiple objects + multiple labels
- Bounding box: center `(x,y)`, width `w`, height `h`
- Scale coordinates to interval `[0,1]`
- Landmark detection: set of point(s), eg, pose detection, AR photo filters
- Sliding window (naive): computationally expensive

### YOLO
1.  For each grid cell: get bounding boxes
2.  Cull low probability boxes
3.  For each target class: non-max suppression -> final predictions
- You Only Look Once
- OverFeat: convert `fc` layers to `conv` -> sliding window in 1 pass
- Split image into grid (eg, `19x19`)
- Bounding box parameters are learned
- Intersection over union (`iou`): measure of bounding box overlap
- Non-max suppresion: detect an object only once (highest `iou` + probability)
- Anchor box: detect multiple objects in 1 grid cell (augment `y_hat`)
    - no. anchor boxes must equal max no. objects in grid cell -> else tiebreak
    - Specialize by shape, but bad at multiple objects with same shape
    - K-means to select anchor boxes that best represent target classes

### Regions with CNN (R-CNN)
- Region proposals: most grid cells don't contain an object
- R-CNN: segmentation algorithm -> blobs -> label + bounding box, 1 at a time
- Fast R-CNN: convolutional sliding window on blobs (instead of 1 at a time)
- Faster R-CNN: use CNN to propose regions (instead of segmentation)
- Still slower than YOLO: 2 steps (region + detect) vs 1 step (detect)


## 2.3 Extras

### Face Recognition & Face Verification
- Verify: `1:1` input image + input name/ID -> yes/no match
- Recognize: `1:N` input image -> output name/ID or `None`
- One shot learning: learn a similarity function (scales with growing database)
- Siamese network: 2 inputs -> same network -> compare 2 outputs
- Triplet loss: `L(A, +, -) = max(0, L2|f(A), f(+)| - L2|f(A), f(-)| + alpha)`
- FaceNet: choose most "difficult" triplets for training
- DeepFace: `(i,j)` -> siamese network -> `d|f(i), f(j)|` -> logistic in `(0,1)`

### Neural Style Transfer
- Gatys et al (2015), *A Neural Algorithm of Artistic Style*
- Content (`C`) + Style (`S`) -> Generated Image (`G`)
- Zeiler & Fergus (2013), *Visualizing and Understanding Convolutional Networks*
- Loss: `J(G) = alpha * J_C (C, G) + beta * J_S (S, G)`
- `J_C (C, G)` = `d|a_(l,C), a_(l,G)|` using pre-trained CNN at layer `l`
- `J_S (S, G)`: weighted sum of L2 norm of style matrices (aka Gram matrices)
    - Style matrix: unnormalized cross covariance -> similarity across channels


--------------------------------------------------------------------------------


# 3 Sequence Models


### Applications
- Speech recognition
- Music generation
- DNA sequence analysis
- Machine translation
- Video labeling
- NLP: sentiment classification, named entity recognition (eg, names, locations)


## 3.1 Architecture

### RNN: Recurrent Neural Network
- Todo

### GRU: Gated Recurrent Unit
- Todo

### LSTM: Long Short-Term Memory
- Todo


--------------------------------------------------------------------------------


# 4 GAN: Generative Adversarial Models


## 4.1

### 4.1.1
- Todo


--------------------------------------------------------------------------------


# 5 Reinforcement Learning


## 5.1

### 5.1.1
- Todo

