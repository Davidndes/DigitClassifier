# Handwritten Digit Recognition with a CNN

The goal of this personal project is to design a CNN capable of recognizing handwritten digits. As one of my first projects I have approached writing this readme in a way that not only makes the experiment easily replicable but also documents the design choices, things I learned and questions I had as someone with a ChemE / Finance background relatively new to the ML space.

# Quick facts
1. Dataset: MNIST — 60,000 training images and 10,000 test images of handwritten digits, each 28×28 grayscale pixels with the correct label
2. Model: Convolutional Neural Network (CNN), built in PyTorch
3. Architecture: convolution → ReLU → max pooling (repeated), then flatten → fully connected layer → 10-way prediction
4. Loss: cross-entropy
5. Optimizers: SGD or Adam (selectable as comparing them was one of the goals)
6. Current accuracy: ~90% on the test set (see "Results and Open Questions" below for why that's lower than it should be)

# How I thought about the problem

To start, the initial question I tried to answer was: If I have 60,000 images of handwritten digits, each with the correct answer, how would I design a program that learns to identify them?

To answer that I asked myself what separates digits from each other? I'd say each is a series of curves, straight lines and circles. Therefore these would be the inputs the model should learn to distinguish in order to be able to classify them. It should learn to associate the presence of certain features with certain digits. This would theoretically allow differentiation between an 8 from a 7, but not necessarily an 8 from a 6.

It should also learn to distinguish what could be described as subfeatures:

The amount of them — for example, 1 typically has one straight line while 7 typically has two ("typically" because people can draw it differently).
The relative sizes of them — using the same 1 and 7 example, in the event someone drew both 1 and 7 with two straight lines, the relative sizes between them will most often be different. i.e. the line at the top of 1 will be relatively shorter compared to the longest line in the digit than in 7.
For curves, one could look at the deviation of the center of a line from a hypothetical straight line between its two end points.

I also wondered whether, after learning to associate these features with certain digits, the model would need some kind of "susceptibility feature." What do I mean by this? Even amongst humans there are sometimes issues making out handwriting. A professor writing equations on the board and his "lambda's" look like "x's". A "susceptibility feature" would take this into account, where the model says "This looks like a seven, but I know that sometimes 1 has looked like seven" so this should be taken into account.

What I learned: the model doesn't need a separate mechanism for this. The final layer outputs a probability for each of the 10 digits, so "70% seven, 25% one" is already the model expressing exactly that uncertainty.

Once the model learns the parameters to make its predictions, a loss function should be used to determine how far off the predictions are from the correct answers. This will be used to calculate the gradients of the parameters via backpropagation. Once the gradients are known, an optimizer (like SGD, Adam etc) uses them to adjust the parameters to reduce that loss, and this process is repeated to improve the predictions.

Can we identify if the above are the "concepts" that the model learns to make predictions? If the model works accurately, can we compare it to another model to see if there are any similarities in the learning described? (I've since learned this is the kind of question mechanistic interpretability tries to answer.)

# How a CNN actually sees it

Refining the proposal, a Convolutional Neural Network (CNN) will be used here. It does not "see" the digits as humans do but as a series of pixels.

Immediately this brings up a question of how will it identify curvature from pixels? How does that translate mathematically? Perhaps through deviation, ie, for a curve vs a straight line the center of it should be the "deepest" part. Where deep can be mathematically described as a range of pixels to the right or left of the start, approached with a gradient. Separately these may exist on an axis with x, y coordinates — so each can actually be represented as a vector. For example, for a straight line moving through (1,1) to (2,2) the gradient should be equal to 1 (well a range around this to account for margin of error). Then for a curve you'll find that as you move along the line this slope will instead continuously change, i.e., the second derivative of the points is non-zero.

What I learned: the CNN is never told to look for curvature, lines or circles. The kernels start out random and learn whatever features reduce the loss. Those features may or may not line up with my human-readable list above (which is exactly what the interpretability question is about).

The computer sees a 28×28 set of pixels as a matrix of numbers where each entry is a number from 0 to 255 for how bright that pixel is (MNIST is grayscale, so there's one channel, not three).

In a CNN, you have a filter (called a kernel) going through the pixels, represented as an m×m matrix. This kernel's entries are the learned parameters (weights, plus a bias). It slides across the pixels of the digit and at each position produces a value based on how well the current area of the image matches the kernel (called an activation). Thus the CNN learns its filters / kernels from the data.

# The pseudocode is below:

1. Train the model on a data set specific for training and have it adjust the initial weights and biases in its kernels. This set would be batches of digits and their correct labels.
2. Combine these kernel outputs (from sliding across the images) to create feature maps.
3. Pass the output from these feature maps to another layer which uses those inputs to further refine the learned weights and biases.
4. Pass these to a final layer where the model makes a prediction among the 10 digits.
5. Compare the prediction with the correct answers using a cross-entropy loss function.
6. Use backpropagation to observe how certain kernel weights affect the loss and update those parameters to reduce the loss corresponding to the incorrect guesses using optimizers.
7. Test on the testing dataset.

A CNN takes images in the tensor format (batch, channels, height, width). A 3D tensor can be thought of as a stack of matrices. The kernels scan over the matrices of the images and produce the feature maps. The outputs of the feature maps are then passed through ReLU, a nonlinearity function which floors the values at zero. This allows the network to represent more complex functions than a stack of purely linear operations could. Afterwards, pooling is done to reduce the number of entries between layers, reducing sensitivity to noise while keeping the largest activations.

Hence the sequence follows convolution → activation → pooling → repeat.

At any stage of each of the processes, the tensors contain x number of numerical values, where x = height × width × number of feature maps. The number of learnable parameters in each layer is (kernel height × kernel width × input channels + 1) × output channels (where the +1 is the bias).

Then for the final layer, the stacks of matrices are flattened into a vector. This is done so that every point from the feature map can be used as an input in a fully connected layer. After this comes tuning (i.e. the backpropagation and optimizers).

In earlier layers, the neurons shared weights but in this connected layer they use different weights. How do they move from sharing weights to having different ones? The switch happens based on the kind of layer you decide to create. The reason we don't have fully connected layers from before is because sharing weights exploits the fact that digits have similar features which means you can save on parameters. How does each output neuron start corresponding to a specific number? We select them and they learn through the loss function, backpropagation and the optimizer. The loss function is what will "teach" the neurons whether their predictions are right or wrong, then comes backpropagation, which will observe how the gradients of the parameters affect the loss — information used by the optimizer to update the parameters.

In this particular experiment I was interested to see how selecting different optimizers might affect the model's accuracy. Thus the model was designed in such a way as to be able to update the parameters with the choice between using SGD and Adam.

# Results and Open Questions

The initial accuracy of the algorithm's predictions is around 90%, giving much room for improvement during a further investigatory phase. A basic CNN on MNIST should get 98–99%, so the gap itself is a question: likely suspects are too few epochs, the learning rate, or not normalizing the inputs.

It was also noticed that among the model's wrong predictions, the most common error was mistaking 3's for 5's. Investigating this further may allow a better understanding of what the model actually "learned" inside its weights and biases.

As I continue to refine this project, further questions also include:

1. Why is accuracy 90% and not 98%? Which of the suspects above is it?
2. Will increasing the layers increase model accuracy? If it does, is there a saturation point?
3. Can a different nonlinearity function perform better here?
4. Does SGD vs Adam change the final accuracy, or just how fast it gets there?
