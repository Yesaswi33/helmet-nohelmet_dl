

https://github.com/user-attachments/assets/437321a0-2152-487a-8f2d-53e1e2318ff0




“I built a hybrid CNN model that combines ResNet50, EfficientNet, and MobileNet for helmet vs no-helmet detection, and deployed it using FastAPI.”

Goal

Binary image classification:

with_helmet

without_helmet

Why hybrid?

Each backbone learns different types of visual features

Combining them improves generalization and robustness

2️⃣ High-Level Architecture (Core Explanation)
🔹 Backbone Networks Used
Model	Strength
ResNet50	Deep semantic features, strong gradient flow
EfficientNet	Optimal depth–width–resolution scaling
MobileNet	Lightweight, fast, edge-friendly features
🔹 Hybrid Design (Conceptual)
Input Image
   ↓
Shared Preprocessing (256×256)
   ↓
ResNet50  → Feature Vector
EfficientNet → Feature Vector
MobileNet → Feature Vector
   ↓
Feature Concatenation
   ↓
Fully Connected Layers
   ↓
Softmax Output (2 classes)


📌 Key idea:
Each model extracts complementary representations, then they’re fused.

3️⃣ Mathematical Intuition (Important for Interviews)
🔹 CNN Feature Extraction

Each backbone learns:

𝑓
𝑖
(
𝑥
)
=
CNN
𝑖
(
𝑥
)
f
i
	​

(x)=CNN
i
	​

(x)

where:

𝑥
x = input image

𝑓
𝑖
f
i
	​

 = feature vector from model 
𝑖
i

🔹 Feature Fusion

Features are concatenated:

𝐹
=
[
𝑓
𝑟
𝑒
𝑠
𝑛
𝑒
𝑡
  
∣
∣
  
𝑓
𝑒
𝑓
𝑓
𝑖
𝑐
𝑖
𝑒
𝑛
𝑡
𝑛
𝑒
𝑡
  
∣
∣
  
𝑓
𝑚
𝑜
𝑏
𝑖
𝑙
𝑒
𝑛
𝑒
𝑡
]
F=[f
resnet
	​

∣∣f
efficientnet
	​

∣∣f
mobilenet
	​

]

This creates a richer representation space.

🔹 Classification Layer

Final dense layer computes:

𝑧
=
𝑊
𝐹
+
𝑏
z=WF+b

Softmax converts logits to probabilities:

𝑃
(
𝑦
=
𝑘
)
=
𝑒
𝑧
𝑘
∑
𝑗
𝑒
𝑧
𝑗
P(y=k)=
∑
j
	​

e
z
j
	​

e
z
k
	​

	​


Binary output:

Helmet

No Helmet

4️⃣ Why Each Model Matters (Strong Interview Point)
🔹 ResNet50 – Deep Understanding

Uses skip connections

𝑦
=
𝐹
(
𝑥
)
+
𝑥
y=F(x)+x

Solves vanishing gradients

Captures global semantic cues like helmet shape

🔹 EfficientNet – Balanced Scaling

Scales depth, width, resolution together

Learns fine-grained textures

Efficient use of parameters

🔹 MobileNet – Speed & Edge Awareness

Uses depthwise separable convolutions

Standard Conv
=
𝐻
𝑊
⋅
𝐶
𝑖
𝑛
⋅
𝐶
𝑜
𝑢
𝑡
Standard Conv=HW⋅C
in
	​

⋅C
out
	​

Depthwise Conv
=
𝐻
𝑊
⋅
𝐶
𝑖
𝑛
Depthwise Conv=HW⋅C
in
	​

Pointwise Conv
=
𝐶
𝑖
𝑛
⋅
𝐶
𝑜
𝑢
𝑡
Pointwise Conv=C
in
	​

⋅C
out
	​


Captures lightweight local features

Makes model deployment-friendly

5️⃣ Why Hybrid > Single Model (Must Say This)

✅ Reduces model bias
✅ Improves feature diversity
✅ Better performance under:

Different lighting

Occlusions

Camera angles

“If one backbone misses a cue, another compensates.”

6️⃣ Training Strategy (Even if Notebook is Lost)

You can confidently say:

Used transfer learning

Loaded pretrained ImageNet weights

Froze early layers initially

Fine-tuned later layers

Optimizer: Adam

𝜃
=
𝜃
−
𝛼
⋅
∇
𝐿
(
𝜃
)
θ=θ−α⋅∇L(θ)

Loss: Categorical Cross-Entropy

𝐿
=
−
∑
𝑦
log
⁡
(
𝑦
^
)
L=−∑ylog(
y
^
	​

)
7️⃣ Preprocessing Pipeline (Your FastAPI Code Matches This)

✔ Resize to 256 × 256
✔ Normalize to [0,1]
✔ Batch dimension added
✔ TensorFlow decoding (framework-consistent)

This is correct and production-ready.

8️⃣ Deployment Architecture (Very Important)
🔹 FastAPI Inference Flow
Client → Image Upload
       → TensorFlow Preprocessing
       → Hybrid Model Prediction
       → Softmax Probability
       → HTML Response

🔹 Why FastAPI?

Async

Lightweight

Production-ready

Easy ML integration

9️⃣ Confidence Score Explanation
confidence
=
max
⁡
(
softmax output
)
×
100
confidence=max(softmax output)×100

Shows model certainty, not just label.
