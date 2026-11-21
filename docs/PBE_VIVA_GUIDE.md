# Project-Based Evaluation (PBE) Viva & Defense Guide

**Project:** Medical Imaging Triage & Lesion Analysis Workstation (`medimg`)  
**Academic Context:** 5th-Semester Evaluation, Undergraduate Engineering  
**Role:** Candidate / Systems & Machine Learning Engineering Lead

This document equips the candidate with mathematically defensible, scientifically rigorous, and architecturally precise answers to the most demanding evaluation questions.

---

## Pillar 1: Mathematical Foundations & Pure NumPy Implementations

### Q1: Why did you implement convolution from scratch in NumPy instead of using PyTorch?
**Defensible Answer:**
"Implementing 2D convolution directly in NumPy using first principles served as our baseline verification anchor. In `src/medimg/fundamentals/conv.py`, we implemented:
$$O[c_{\text{out}}, h, w] = \text{bias}[c_{\text{out}}] + \sum_{c_{\text{in}}} \sum_{kh} \sum_{kw} I[c_{\text{in}}, h \cdot s + kh, w \cdot s + kw] \cdot K[c_{\text{out}}, c_{\text{in}}, kh, kw]$$
This guaranteed that we understand spatial receptive fields, striding arithmetic, padding dimensions, and vector broadcasting before building complex deep models. Furthermore, our unit tests in `tests/test_fundamentals.py` explicitly verify numerical parity within $10^{-6}$ against PyTorch autograd tensors."

### Q2: What is the mathematical limitation of the McCulloch-Pitts neuron, and how does the Sigmoid neuron resolve it?
**Defensible Answer:**
"A McCulloch-Pitts neuron uses a non-differentiable step activation function $\Theta(z) = \mathbf{1}_{\{z \ge \theta\}}$ with binary thresholding. Because its derivative is zero everywhere (except at $z=0$ where it is undefined), $\frac{\partial \mathcal{L}}{\partial w} = 0$, meaning gradient-based backpropagation cannot train the weights.  
In contrast, the sigmoid neuron $\sigma(z) = \frac{1}{1 + e^{-z}}$ provides a smooth, non-linear activation with a non-zero analytical derivative:
$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$
This allows the chain rule of calculus to compute exact error gradients during backpropagation. However, both single McCulloch-Pitts and single sigmoid perceptrons cannot solve linearly non-separable problems such as XOR without hidden layers."

### Q3: Why does zero-initialization fail in deep multi-layer neural networks?
**Defensible Answer:**
"If all weights in a hidden layer are initialized to zero ($W^{(l)} = \mathbf{0}$), every hidden unit receives the exact same pre-activation ($z_j = \mathbf{0}$), produces the exact same activation ($a_j = \sigma(0) = 0.5$), and propagates the exact same gradient ($\frac{\partial \mathcal{L}}{\partial z_j}$) backwards. Because $\Delta W_{jk} = \eta \delta_j a_k$, all weights in the layer update identically. The neurons remain symmetric throughout training, collapsing the representation capacity to that of a single neuron. This is proven mathematically in `test_zero_init_symmetry`."

### Q4: What is the difference between Hebbian Learning and Oja's Rule?
**Defensible Answer:**
"Classical Hebbian learning updates synaptic weights according to $\Delta w = \eta y x$. If the inputs and outputs are positively correlated, the weight vector grows exponentially without bound ($\|w\| \to \infty$).  
Oja's rule introduces an activity-dependent normalization weight decay term:
$$\Delta w = \eta y (x - y w)$$
Mathematically, Oja's rule converges to $\|w\|_2 = 1$ and extracts the first principal component (eigenvector corresponding to the largest eigenvalue of the data covariance matrix), which we verified empirically against SVD in `test_hebb_unbounded_oja_matches_pca`."

---

## Pillar 2: Dataset Strategy & Grouping Semantics

### Q5: How did you split the HAM10000 dataset, and why is random train/test splitting scientifically invalid?
**Defensible Answer:**
"In dermatological datasets, multiple dermoscopy images often depict the same underlying skin lesion from different angles, lighting, or follow-ups. In HAM10000, 10,015 images represent only 7,470 unique lesions.  
If you perform naive random splitting, images of the same lesion appear in both the training and test sets. The model simply memorizes lesion-specific features (such as border eccentricities, hairs, or biopsy marks), producing artificially inflated test accuracy.  
To prevent this, we performed **Group Stratified Splitting** grouped strictly on `lesion_id` using a 70/15/15 ratio. Furthermore, we honestly document that HAM10000 metadata does not release `patient_id` due to anonymization, so our split guarantees zero lesion-level leakage, with a small residual risk if a patient had multiple distinct lesions."

### Q6: The melanoma class is heavily imbalanced (~11% positive). How did you handle this during training?
**Defensible Answer:**
"We employed a dual-strategy approach:
1. **Class-Weighted Loss:** We parameterized `nn.BCEWithLogitsLoss(pos_weight=pos_weight)`, where $\text{pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 3.67$. This penalizes false negatives $3.67\times$ more heavily than false positives.
2. **Weighted Random Sampling:** In minority oversampling runs, we applied `WeightedRandomSampler` to present batches with balanced class proportions without duplicating disk assets."

---

## Pillar 3: Vision Architectures & Objective Functions

### Q7: Why did you select ResNet-18 as your primary production classifier over deeper architectures like ResNet-50 or GoogLeNet?
**Defensible Answer:**
"We followed empirical evidence from our benchmark evaluations:
- ResNet-18 achieved held-out **AUROC = 0.8953**, compared to 0.8872 for ResNet-50 and 0.8654 for GoogLeNet.
- ResNet-18 contains 11.19M parameters, whereas ResNet-50 contains 25.56M. Given HAM10000's finite size, ResNet-50 exhibited slight overfitting.
- ResNet-18 executed in **16.2 ms on CPU** and **7.15 ms in ONNX Runtime**, comfortably meeting our interactive clinical latency SLA."

### Q8: What loss function did you use for RetinaNet, and why does standard cross-entropy fail for dense object detectors?
**Defensible Answer:**
"Standard cross-entropy fails in one-stage dense object detectors like RetinaNet because the vast majority of anchor boxes (thousands per image) contain easy background negatives. Their cumulative small gradients overwhelm the rare foreground lesion gradients.  
We used **Focal Loss**:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Setting $\alpha = 0.25$ balances foreground vs background, and $\gamma = 2.0$ dynamically down-weights easy background examples ($p_t \to 1 \implies (1 - p_t)^\gamma \to 0$), forcing the network to concentrate gradients on hard, ambiguous lesion candidates."

---

## Pillar 4: Probability Calibration & Clinical Operating Thresholds

### Q9: Why is temperature scaling necessary, and how does it affect AUROC?
**Defensible Answer:**
"Deep modern neural networks trained with cross-entropy and weight decay are notoriously overconfident: a model outputting $0.90$ confidence is typically correct only $70\%$ of the time.  
Temperature scaling applies a single learned parameter $T > 0$ to scale logits:
$$p_{\text{cal}} = \sigma\left(\frac{z}{T}\right)$$
Because dividing by a constant $T$ is a strictly monotonic transformation:
$$\text{rank}(z_i) = \text{rank}\left(\frac{z_i}{T}\right)$$
It leaves the ROC curve, PR curve, and AUROC **completely unchanged**! What it changes is the **calibration error**: it optimizes the Negative Log-Likelihood on held-out validation data via L-BFGS, reducing Expected Calibration Error (ECE) and Brier score so predicted probabilities reflect true empirical clinical risk."

### Q10: How did you select the triage decision threshold ($\theta = 0.518$)?
**Defensible Answer:**
"In medical triage, missing a malignant melanoma (False Negative) is clinically catastrophic, whereas unnecessary review of a benign nevus (False Positive) is merely inconvenient.  
Therefore, the threshold $\theta = 0.518$ was selected **strictly on the validation set** to achieve:
$$\text{Sensitivity}(\theta) = \frac{\text{TP}}{\text{TP} + \text{FN}} \ge 95\%$$
This threshold was frozen and then evaluated strictly once on the held-out test cohort, where it confirmed high sensitivity."

---

## Pillar 5: Explainability (Grad-CAM) & Failure Modes

### Q11: How does Grad-CAM work mathematically, and what are its clinical limitations?
**Defensible Answer:**
"Grad-CAM computes the importance of the $k$-th feature map $A^k$ in the final convolutional layer (`layer4[-1]`) for class $c$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
The activation map is then a rectified linear combination:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
**Limitations:**
1. Low spatial resolution: The final conv layer in ResNet-18 has spatial dimensions $4 \times 4$ for a $128 \times 128$ input; upsampling to native resolution introduces bilinear blur.
2. Visual correlation is not clinical causation: Grad-CAM highlights features the network attended to, which may include confounding artifacts like dermoscopy gel bubbles, ruler marks, or dark hairs."

---

## Pillar 6: Software Engineering & Remediation Highlights

### Q12: What was the PyTorch CPU state-dict aliasing bug, and how did you resolve it?
**Defensible Answer:**
"In PyTorch on CPU, calling `v.cpu()` on a tensor that is already on CPU returns the exact same tensor reference sharing the underlying storage. Saving `best_state = {k: v.cpu() for k, v in model.state_dict().items()}` created a shallow reference. In subsequent epochs, gradient updates mutated `best_state` in-place!  
We resolved this by implementing `clone_state_dict` in `src/medimg/checkpoints.py`:
```python
return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
```
This guarantees an isolated, immutable deep copy."

### Q13: How did you verify ONNX Runtime parity and speedup?
**Defensible Answer:**
"In `scripts/aggressive_model_check.py` and `tests/test_torch_vision.py`, we evaluated 25 real dermoscopy images simultaneously through PyTorch and ONNX Runtime. The maximum absolute logit difference was **$3.34 \times 10^{-6}$**, well within our strict $10^{-4}$ tolerance. ONNX Runtime delivered a **$2.33\times$ speedup** on CPU (reducing inference latency from 16.2 ms to 6.95 ms)."
