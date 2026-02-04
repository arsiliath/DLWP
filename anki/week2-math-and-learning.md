# Week 2 — Mathematical Foundations & Learning

Q: What does a dense layer compute?
A: output = activation(W @ x + b). It applies a linear transformation (weight matrix times input plus bias), then a nonlinear activation function. W has shape (input_dim, units), b has shape (units,).

---

Q: How many parameters does a Dense layer with 64 inputs and 32 units have?
A: (64 × 32) + 32 = 2,080. The weight matrix has input_dim × units parameters, plus one bias per unit.

---

Q: Why do we need nonlinear activation functions?
A: Without them, stacking layers just produces another linear transformation (a matrix times a matrix is still a matrix). Nonlinearities let the network learn non-linear decision boundaries — they "fold" the data space in ways that linear transforms cannot.

---

Q: What is a gradient?
A: The gradient of a function is the vector of its partial derivatives. It points in the direction of steepest ascent. In deep learning, we move weights in the *opposite* direction (steepest descent) to minimize the loss.

---

Q: What is backpropagation?
A: The modular application of the chain rule to compute gradients layer by layer, from output back to input. Each operation knows its own local gradient. The chain rule multiplies these local gradients together to get the gradient of the loss with respect to any weight in the network.

---

Q: Why can't we initialize all weights to zero?
A: It breaks symmetry. If all weights are identical, every neuron in a layer computes the same thing, receives the same gradient, and updates identically. The network effectively has one neuron per layer. Random initialization ensures neurons learn different features.

---

Q: What does the learning rate control?
A: The size of each weight update step. Too small: training is very slow and may get stuck. Too large: updates overshoot the minimum and loss diverges. It scales the gradient: new_weight = old_weight - learning_rate × gradient.

---

Q: What is SGD with momentum?
A: Instead of updating weights using only the current gradient, momentum maintains an exponential moving average of past gradients. This smooths out noisy updates and helps the optimizer move through narrow valleys without oscillating.

---

Q: Why can't you use accuracy as a loss function?
A: Accuracy is a step function — a prediction is either right or wrong. It is not differentiable, so the gradient is zero almost everywhere. The optimizer gets no signal about *which direction* to adjust weights. Loss functions like cross-entropy are smooth and differentiable, so every small weight change produces a measurable change in loss.

---

Q: What does normalization of input data do and why is it needed?
A: It rescales features to small values centered near 0 (e.g., zero mean, unit variance). Two reasons: (1) Features on different scales create an elongated loss landscape, causing gradient descent to zigzag inefficiently. Normalization makes the landscape rounder. (2) Network internals (activations, weight initialization) are designed for small input values — large values cause gradient explosion or saturation.

---

Q: What is broadcasting?
A: When operating on tensors of different shapes, the smaller tensor is virtually "stretched" to match the larger one. Rules: axes are compared right-to-left; dimensions must either match or one must be 1 (which gets broadcast). No data is actually copied — it's handled efficiently by the framework.
