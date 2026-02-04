# Week 3 — Classification & The ML Workflow

Q: What output activation and loss do you use for binary classification?
A: **Sigmoid** activation (outputs a single value between 0 and 1) with **binary cross-entropy** loss. Sigmoid squashes any real number into a probability.

---

Q: What is binary cross-entropy intuitively?
A: It measures how far your predicted probability is from the true label (0 or 1). Predicting 0.99 when the true label is 1 gives very low loss. Predicting 0.01 when the true label is 1 gives very high loss. It penalizes confident wrong predictions harshly.

---

Q: What is multi-hot encoding vs one-hot encoding?
A: **One-hot**: a vector with exactly one 1 and the rest 0s — represents a single category (e.g., [0, 0, 1, 0] = class 2). **Multi-hot**: a vector with potentially many 1s — represents presence/absence of multiple features (e.g., [1, 0, 1, 1, 0] = words 0, 2, and 3 appear in this text).

---

Q: Why do we split data into train, validation, and test sets?
A: **Training set**: the model learns from this. **Validation set**: used to tune hyperparameters and decide when to stop training. **Test set**: touched only once at the very end to estimate real-world performance. If you tune on the test set, you overfit to it and your performance estimate is unreliable.

---

Q: What does overfitting look like on a loss curve?
A: Training loss keeps decreasing while validation loss starts increasing. The gap between them widens. This means the model is memorizing training-specific patterns (noise, quirks) rather than learning generalizable features.

---

Q: What is a commonsense baseline and why do you need one?
A: The simplest possible predictor — e.g., always guessing the most common class, or random guessing. For balanced binary classification, random gives 50% accuracy. If your neural network only gets 52%, it's barely better than flipping a coin. The baseline tells you whether your model is actually learning anything meaningful.

---

Q: What is the early stopping workflow?
A: 1. Train with a validation split and watch validation loss. 2. Note the epoch where validation loss is lowest (before it starts rising). 3. Retrain from scratch on the full training set (train + validation) for exactly that many epochs. 4. Evaluate once on the held-out test set.

---

Q: Why do you retrain on the full dataset after finding the best epoch?
A: Because during hyperparameter tuning you held out validation data, so the model only trained on a subset. Once you know the right number of epochs, you want to give the model all available training data for that final run to get the best possible performance.
