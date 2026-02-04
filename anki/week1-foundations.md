# Week 1 — Foundations

Q: What are the three components of a neural network training loop?
A: 1. **Layers** transform input data into predictions. 2. A **loss function** measures how far predictions are from targets. 3. An **optimizer** adjusts weights using gradients to minimize the loss.

---

Q: What does "deep" mean in deep learning?
A: Depth refers to the number of layers (3+ = deep). Each layer learns an intermediate representation of the data, feeding into the next. The network learns a chain of transformations, not a single mapping.

---

Q: Why is deep learning called "representation learning"?
A: Instead of hand-engineering features (like edge detectors for images), the network learns its own internal representations of the data — each layer transforms the data into a form that makes the task easier for subsequent layers.

---

Q: What is a tensor?
A: A container for numerical data. Scalars are rank-0 tensors, vectors are rank-1, matrices are rank-2, and higher-dimensional arrays are rank-3+. Tensors are described by their rank (number of axes), shape (size along each axis), and dtype (data type).

---

Q: What are the three key attributes of a tensor?
A: 1. **Rank** (ndim) — number of axes. 2. **Shape** — tuple of integers giving size along each axis. 3. **Dtype** — the data type (float32, int32, etc.).

---

Q: What shape is a batch of 128 color images at 256×256 resolution?
A: (128, 256, 256, 3) — batch size, height, width, channels (RGB).
