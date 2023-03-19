### Curse of Dimensionality in ANN
- More connections
- More weights to train
- Longer training

### Components of CNN
- Convolution
    - Extracts features from image
    - Preserves feature spatial relationships
        - Edges
        - Composite elements (nose, eye)
    - Reduced computation
    - Hyperparameters:
        - Kernel size
            - related pixels
        - Number of filters
            - Feature detected
        - Stride
            - Distance to move filter
            - Larger stride -> pixel independence
            - Larger values faster
                - Decrease size of feature map
                - Reduces information passed to next layer
            - 1 is common value
        - Padding
- Filters
    - Values not fixed
    - Values are what we train
    - Improved through training
    - Trained on labeled images
    - Detect unique features that determine objects
    - CNN training faster since only filter weights trained
    - Weights shared across image
- Non-linearity (AF: ReLU)
    - Lets NN handle non-linear
    - Added in two ways
        - As a layer after convolution layer
        - As parameter to convolution layer
    - ReLU most common non-linearity function
        - y = max(0, x)
        - If (x < 0) return 0 else return x
    - Prevents vanishing gradient
- Pooling
    - Reduce Dimensionality
    - Translational Invariance
    
- Classification    

### Issues with Convolution
- Reduction in spatial dimensions
- Data at edges is used less
- 