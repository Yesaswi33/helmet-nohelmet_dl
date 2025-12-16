

https://github.com/user-attachments/assets/437321a0-2152-487a-8f2d-53e1e2318ff0




 ## Hybrid CNN Helmet Detection System

### Overview

This project implements a **hybrid deep learning model** for **binary image classification** to detect whether a person is **wearing a helmet or not**.
The model combines multiple CNN backbones to improve robustness and generalization and is deployed using **FastAPI** for real-time inference.

**Classes**

* `with_helmet`
* `without_helmet`

---

## Problem Statement

Helmet detection systems must perform reliably under varying conditions such as:

* Different lighting environments
* Occlusions
* Camera angles
* Image quality variations

A single CNN backbone often struggles to generalize across all scenarios.
To address this, a **hybrid architecture** is used.

---

## Why a Hybrid Model?

Each CNN backbone learns **different visual representations**.
By combining them, the model gains complementary strengths.

**Key Advantages**

* Reduced model bias
* Improved feature diversity
* Better performance in real-world conditions

> If one backbone misses a visual cue, another compensates.

---

## High-Level Architecture

**Processing Flow**

Input Image
→ Shared Preprocessing (256 × 256)
→ ResNet50 Feature Extraction
→ EfficientNet Feature Extraction
→ MobileNet Feature Extraction
→ Feature Concatenation
→ Fully Connected Layers
→ Softmax Output (2 Classes)

**Core Idea**
All three backbones extract features independently.
Their feature vectors are fused to create a richer representation before classification.

---

## Backbone Networks Used

**ResNet50**

* Deep semantic feature extraction
* Skip connections ensure strong gradient flow
* Captures global structures such as helmet shape

**EfficientNet**

* Balanced scaling of depth, width, and resolution
* Learns fine-grained textures
* High accuracy with efficient parameter usage

**MobileNet**

* Lightweight and fast
* Uses depthwise separable convolutions
* Captures local features and supports deployment efficiency

---

## Mathematical Intuition

### Feature Extraction

Each CNN backbone learns a function:

fᵢ(x) = CNNᵢ(x)

Where:

* x is the input image
* fᵢ is the feature vector extracted by model i

---

### Feature Fusion

The extracted features are concatenated:

F = [f_resnet || f_efficientnet || f_mobilenet]

This fusion creates a richer and more expressive feature space.

---

### Classification

The final dense layer computes:

z = W · F + b

Softmax converts logits into probabilities:

P(y = k) = eᶻᵏ / Σⱼ eᶻⱼ

**Output**

* Helmet
* No Helmet

---

## Why Each Model Matters

**ResNet50**

* Uses residual connections: y = F(x) + x
* Prevents vanishing gradients
* Strong at capturing global semantic cues

**EfficientNet**

* Compound scaling strategy
* Extracts detailed texture information
* Maintains performance with fewer parameters

**MobileNet**

* Uses depthwise + pointwise convolutions
* Reduces computational cost
* Enables fast and deployment-friendly inference

---

## Training Strategy

Even without the original notebook, the training pipeline is well-defined.

**Approach**

* Transfer learning using ImageNet pretrained weights
* Early layers frozen initially
* Later layers fine-tuned

**Optimizer**

* Adam optimizer

**Weight Update Rule**
θ = θ − α · ∇L(θ)

**Loss Function**

* Categorical Cross-Entropy

---

## Preprocessing Pipeline

The preprocessing pipeline used during inference matches training conditions.

**Steps**

* Resize image to 256 × 256
* Normalize pixel values to [0, 1]
* Add batch dimension
* TensorFlow-based decoding for consistency

This ensures reliable and production-ready predictions.

---

## Deployment Architecture

**Inference Flow**
Client uploads image
→ FastAPI endpoint
→ TensorFlow preprocessing
→ Hybrid CNN prediction
→ Softmax probability
→ HTML response with label and confidence

---

## Why FastAPI?

* Asynchronous and lightweight
* Production-ready
* Easy integration with deep learning models
* Fast inference response times

---

## Confidence Score

Model confidence is calculated as:

confidence = max(softmax output) × 100

This represents how certain the model is about its prediction, not just the predicted class.

---

## Summary

This hybrid CNN approach combines **deep semantic understanding**, **fine-grained texture learning**, and **lightweight feature extraction** into a single robust system.
The result is a reliable, scalable, and deployment-ready helmet detection model suitable for real-world applications.
