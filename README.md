# Image Analysis & Artificial Vision

## Assignment 1 — Image Rectification Toolkit

**Goal:** Design and implement a computer vision pipeline in Python (NumPy + OpenCV) that rectifies perspective-distorted photographs of flat rectangular objects (e.g., book covers, posters, documents) into a clean, front-facing view.

### What's implemented

- **Convolution & Filtering** — Custom 2D convolution from scratch (zero / replicate / reflect padding), mean filter, Gaussian filter, and three noise models (Gaussian, impulse/salt-and-pepper, Poisson). Color handling via luminance-only (HSV) vs. per-channel RGB filtering.
- **Canny Edge Detection** — Full pipeline: Gaussian smoothing → Sobel gradients → Non-Maximum Suppression (NMS) → double thresholding → hysteresis. Compared against `cv2.Canny`.
- **Corner Detection** — Moravec and Harris detectors implemented from scratch with 2D NMS. Multi-scale Gaussian pyramid analysis (3 levels) showing how corner distributions change across scales.
- **Projective Rectification** *(optional bonus)* — Harris corners are used to automatically select 4 vertices of the target object. A homography is computed via `cv2.getPerspectiveTransform` and applied with `cv2.warpPerspective` to produce the rectified image. Includes sensitivity analysis for corner perturbations of ±2, ±5, ±10 pixels.

### Key results
- Gaussian filter (σ=1.0) works best for Gaussian and Poisson noise; mean filter (5×5) handles impulse noise better.
- Luminance-only filtering (HSV) preserves colors better than per-channel RGB.
- Harris detector is more sensitive than Moravec but requires careful parameter tuning.
- Rectification quality degrades noticeably with corner errors above ±5 pixels.

### Constraints
Core operators (convolution, edge detection, corner detection) are implemented with NumPy only. OpenCV is used exclusively for image I/O, color conversions, and geometric transforms.

---

## Assignment 2 — Pet Image Classification

**Goal:** Classify pet breed images into 6 classes (Abyssinian, Bengal, Birman, Boxer, English Cocker Spaniel, German Shorthaired) using both a neural and a traditional computer vision pipeline. Dataset: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).

### What's implemented

#### (a) Neural Pipeline — U-Net CNN
- Encoder–decoder U-Net with skip connections. Encoder: 4 levels (32→64→128→256 channels) + 512-channel bottleneck.
- **Dual-head training:** segmentation head (2-class foreground/background) + classification head (6-class breed), trained jointly with a combined loss λ·CE_seg + CE_cls.
- Boundary pixels (`ignore_index=255`) excluded from segmentation loss and evaluation.
- Regularization: weight decay, ReduceLROnPlateau scheduler, AMP mixed precision.
- **Test results:** mIoU ≈ 0.769, CCR ≈ 0.570

#### (b) Traditional Pipeline — Bag-of-Features + MLP
- SIFT feature extraction (`cv2.SIFT_create`) + random sampling of M descriptors.
- K-means visual vocabulary (K ∈ {300, 400, 500}) built manually with `cv2.kmeans`.
- Per-image BoF histograms with L1 normalization.
- Shallow MLP classifier (2 hidden layers: 256, 128 neurons), dropout, weight decay.
- Optimal K selected via validation accuracy.
- **Test results (best K=300):** CCR ≈ 0.338

#### (c) Background Replacement Verification
- Test set backgrounds replaced with low-frequency random noise (using ground-truth segmentation masks and bitwise ops).
- CNN: 0.5695 → 0.5360 CCR; MLP: 0.3384 → 0.3300 CCR. Both models show only a small drop, confirming they primarily rely on the animal rather than the background.

#### (d) Mask-Guided Visual Vocabulary *(optional bonus)*
- BoF vocabulary rebuilt using only SIFT descriptors that fall within foreground masks.
- Tested two histogram construction strategies (all SIFT vs. foreground-only SIFT for training images).

### Constraints
All evaluation metrics (CCR, confusion matrix, mIoU) are implemented manually. No scikit-learn, torchmetrics, or pre-trained models used.

---

## Environment

Both notebooks run on **Google Colab** from top to bottom without manual intervention.

```
Python 3.x
numpy
matplotlib
opencv-python (cv2)
torch + torchvision   # Assignment 2 only
```
