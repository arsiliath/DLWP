# Week 4 — Multiclass Classification & Architecture Decisions

Q: What output activation and loss do you use for multiclass single-label classification?
A: **Softmax** activation (outputs a probability distribution that sums to 1) with **categorical cross-entropy** loss. Each output represents the probability of one class, and exactly one class is correct.

---

Q: Why use softmax instead of sigmoid for multiclass classification?
A: Sigmoid treats each output independently — you could get 0.9 for class A and 0.8 for class B simultaneously. For mutually exclusive classes, that's nonsensical. Softmax enforces competition: increasing one class's probability necessarily decreases the others, because all outputs sum to 1. It produces a proper probability distribution over mutually exclusive outcomes.

---

Q: What is an information bottleneck in a neural network?
A: When an intermediate layer has fewer units than the information needed to solve the task. Example: a 64→4→46 network tries to compress all information through 4 units, then reconstruct 46 class distinctions. The 4-unit layer can only represent ~4 dimensions of variation, which is not enough to distinguish 46 classes. Information is permanently lost.

---

Q: How do you choose the number of units in hidden layers?
A: Hidden layers should generally not have fewer units than the output layer — otherwise you create an information bottleneck. Beyond that, more units = more capacity to learn complex patterns, but also more risk of overfitting. Start simple and increase capacity until validation loss stops improving.

---

Q: What is the complete decision table for output layer configuration?

| Problem | Output activation | Loss function |
|---|---|---|
| Binary classification | sigmoid (1 unit) | binary_crossentropy |
| Multiclass single-label | softmax (N units) | categorical_crossentropy |
| Multiclass multilabel | sigmoid (N units) | binary_crossentropy |
| Regression | none (1 unit) | mse |

---

Q: What is categorical cross-entropy?
A: It measures how far a predicted probability distribution is from the true one-hot distribution. If the true class is 3, it only looks at the predicted probability for class 3 and penalizes based on -log(p). High confidence on the right class → low loss. Low confidence on the right class → high loss.

---

Q: What is the difference between categorical_crossentropy and sparse_categorical_crossentropy?
A: They compute the same thing. The difference is label format: **categorical** expects one-hot encoded labels (e.g., [0, 0, 1, 0]). **Sparse** expects integer labels (e.g., 2). Sparse is more memory-efficient when you have many classes.

---

Q: What makes a machine learning problem learnable?
A: Two conditions: (1) The inputs contain sufficient information to predict the outputs — you can't predict stock prices from weather data. (2) A learnable, smooth mapping exists between inputs and outputs — similar inputs should produce similar outputs, so gradient-based optimization can find the pattern.

---

Q: What is the manifold hypothesis?
A: Real-world high-dimensional data (images, text, audio) doesn't fill the entire high-dimensional space uniformly. Instead, it lies on or near a much lower-dimensional surface (manifold). A 256×256 image has 196,608 dimensions, but the set of "natural images" is a tiny subspace. Neural networks learn to navigate this manifold.

---

Q: Why does a model trained on one dataset sometimes fail on new data even if the task is the same?
A: Sampling bias / distribution shift. If training data isn't representative of real-world deployment conditions (different demographics, lighting, language, time period), the model learns patterns specific to the training distribution. Also, concept drift: the real world changes over time, but the model is frozen at training time.
