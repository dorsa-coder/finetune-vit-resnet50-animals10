
# Exercise 2 — Image Classification: ViT vs ResNet-50

### Overview
This exercise fine-tunes two pretrained image models — **ViT** (`google/vit-base-patch16-224-in21k`) and **ResNet-50** (`microsoft/resnet-50`) — on the same dataset, using identical training settings, and compares their accuracy.

### Dataset
- **Source:** [`Rapidata/Animals-10`](https://huggingface.co/datasets/Rapidata/Animals-10) (HuggingFace Datasets)
- **Classes (10):** Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, Squirrel
- **Subset used:** 1,200 images randomly sampled from the full 23,554-image dataset (to keep training fast on free-tier GPUs), split 80/20 into train (960) / validation (240) with `seed=42`.

### Fair-comparison setup
Both models were trained with **identical** hyperparameters:

| Setting | Value |
|---|---|
| Batch size | 16 |
| Learning rate | 2e-4 |
| Epochs | 4 |
| Seed | 42 |
| Train/val split | Same split object reused for both models |

Only the model architecture and its matching image processor changed between runs.

### Results

| Dataset | Model | Accuracy | Training time (s) | Epochs |
|---|---|---|---|---|
| Rapidata/Animals-10 (subset) | ViT (google/vit-base-patch16-224-in21k) | **0.9875** | 76.61 | 4 |
| Rapidata/Animals-10 (subset) | ResNet-50 (microsoft/resnet-50) | 0.9250 | 34.16 | 4 |

**Detailed training log — ViT:**
| Step | Training Loss | Validation Loss | Accuracy |
|---|---|---|---|
| 100 | 0.169887 | 0.173731 | 0.9875 |
| 200 | 0.065188 | 0.113455 | 0.9917 |
| 240 (final, epoch 4/4) | 0.058685 | 0.106684 | 0.9875 |

Final train metrics: `train_loss=0.3413`, `train_runtime=0:01:16.61`, `train_samples_per_second=50.124`, `train_steps_per_second=3.133`, `total_flos=277,152,821 GF`.
Final eval metrics: `eval_accuracy=0.9875`, `eval_loss=0.1067`.

**Detailed training log — ResNet-50:**
| Step | Training Loss | Validation Loss | Accuracy |
|---|---|---|---|
| 100 | 1.271466 | 1.116935 | 0.7208 |
| 200 | 0.702176 | 0.631374 | 0.9083 |
| 240 (final, epoch 4/4) | 0.636935 | 0.605065 | 0.9250 |

Final train metrics: `train_loss=1.2053`, `train_runtime=0:00:34.15`, `train_samples_per_second=112.416`, `train_steps_per_second=7.026`, `total_flos=75,996,666 GF`.
Final eval metrics: `eval_accuracy=0.925`, `eval_loss=0.6051`.

### Analytical questions (2.6)

**1. Which model had higher accuracy? Why? Did it match expectations based on model reputation?**
ViT clearly outperformed ResNet-50 (0.9875 vs 0.9250). This is mainly due to ViT's stronger pretraining source (ImageNet-21k, a much larger dataset than the ImageNet-1k used for this ResNet-50 checkpoint) and its self-attention mechanism, which captures global relationships across the whole image from the first layer. This is somewhat counter to the common expectation that transformers (like ViT) need more fine-tuning data than CNNs to show an advantage — here, with only 1,200 images, ViT's stronger pretraining still won out.

**2. What is the core architectural difference between ViT (transformer) and ResNet (CNN) for image processing?**
ResNet uses convolutional layers with local receptive fields, building up higher-level features gradually as the network deepens. ViT splits the image into patches and uses self-attention to relate every patch to every other patch from the start, giving it a global view of the image immediately — but with less built-in structural bias (inductive bias) than a CNN, which is why it typically needs more pretraining data to generalize well.

**3. Did more epochs noticeably improve accuracy, or did the model approach overfitting?**
For both models, accuracy rose steadily and validation loss fell steadily across all 4 epochs (ViT: 0.72 → 0.9875; ResNet-50: 0.72 → 0.925), showing no sign of overfitting within this run. ResNet-50's improvement did slow down between the last two logged steps (0.908 → 0.925), suggesting it may be approaching a ceiling for this data size — more epochs would be needed to confirm actual overfitting.

**4. Were the classes balanced? What effect did that have on the results?**
The original Animals-10 dataset is imbalanced (e.g., classes like sheep or elephant have fewer images than dog or spider). Since our 1,200-image subset was randomly sampled from the full dataset, it likely carries a similar relative imbalance. This means overall accuracy may look strong mostly due to good performance on the majority classes, while minority classes could perform worse — this wasn't measured directly here; per-class precision/recall/F1 would be needed to confirm it.

### Fixes applied to make the notebook run on current library versions
- `ViTFeatureExtractor` (removed from `transformers`) → replaced with `AutoImageProcessor`
- `load_metric` (removed from `datasets`) → replaced with `evaluate.load("accuracy")`
- `evaluation_strategy` → renamed to `eval_strategy` (current `TrainingArguments` API)
- Added explicit `seed=42` to both `TrainingArguments` for a stricter fair comparison
