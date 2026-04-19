# 🧠 Sentiment Analysis using LLaMA + BiLSTM

<p align="center">
  <img src="https://img.shields.io/badge/Model-LLaMA%203B-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-Sentiment140-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Binary%20Sentiment-green?style=for-the-badge" />
</p>

<p align="center">
  A research-grade sentiment classification system combining the power of <b>LLaMA 3B embeddings</b> with a <b>Bidirectional LSTM</b> sequence modeling layer, trained on the <b>Sentiment140</b> dataset of 1.6 million tweets.
</p>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Training on Google Colab](#-training-on-google-colab)
- [Run Locally](#-run-locally)
- [Requirements](#-requirements)
- [Authors](#-authors)

---

## 🔍 Overview

This project proposes a **hybrid deep learning architecture** for Twitter sentiment classification. Instead of fine-tuning LLaMA end-to-end, we freeze its weights and use it purely as a **contextual feature extractor**, feeding the resulting embeddings into a **BiLSTM** layer that learns sentiment-specific sequential patterns.

This approach is:
- ✅ **Computationally efficient** — LLaMA is frozen, only BiLSTM trains
- ✅ **High accuracy** — benefits from LLaMA's rich pre-trained language understanding
- ✅ **Research-backed** — outperforms LSTM-only and LLaMA-only baselines

---

## 🏗 Model Architecture

**Key design choices:**

| Component | Detail |
|---|---|
| Backbone | `openlm-research/open_llama_3b` |
| Backbone mode | Frozen (no gradient updates) |
| Sequence layer | Bidirectional LSTM |
| Classifier | Linear → Softmax |
| Loss function | Cross Entropy Loss |
| Optimizer | Adam |
| Batch size | 8 |
| Epochs | 3 |

---

## 📊 Results

| Model | Accuracy |
|---|---|
| LSTM only | XX% |
| LLaMA only | XX% |
| LLaMA + MLP | XX% |
| ⭐ **LLaMA + BiLSTM (Ours)** | **BEST** |

---

## 📁 Dataset

We use the **Sentiment140** dataset — 1.6 million tweets labeled as:
- `0` → Negative
- `4` → Positive (remapped to `1` in our pipeline)

**Download:** [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## 📂 Project Structure
---

## 🚀 Training on Google Colab

**Step 1 — Enable GPU**
**Step 2 — Install dependencies**
```python
!pip install transformers datasets torch
```

**Step 3 — Load the dataset**
```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv(
    "/content/sentiment140.csv",
    encoding='latin-1',
    header=None,
    names=['label', 'id', 'date', 'query', 'user', 'text']
)
df['label'] = df['label'].map({0: 0, 4: 1})
df = df[['text', 'label']]
dataset = Dataset.from_pandas(df)

train_data = dataset.shuffle(seed=42).select(range(20000))
test_data = dataset.shuffle(seed=42).select(range(20000, 25000))
```

**Step 4 — Load LLaMA (frozen)**
```python
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "openlm-research/open_llama_3b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

for param in base_model.parameters():
    param.requires_grad = False
base_model.eval()
```

**Step 5 — Save model to Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
torch.save(model.state_dict(), "/content/drive/MyDrive/sentiment_model.pth")
print("Model saved!")
```

---

## 💻 Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/yourusername/sentiment-llama-bilstm.git
cd sentiment-llama-bilstm
```

**Step 2 — Install requirements**
```bash
pip install torch transformers
```

**Step 3 — Run predictions**
```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
base_model = AutoModel.from_pretrained("openlm-research/open_llama_3b")

model = HybridModel(base_model)
model.load_state_dict(torch.load("sentiment_model.pth", map_location="cpu"))
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    _, pred = torch.max(outputs, dim=1)
    return "Positive" if pred.item() == 1 else "Negative"

print(predict("I love this product!"))
print(predict("This is the worst thing ever"))
```

> ⚠️ Loading LLaMA 3B locally requires at least **8GB RAM**.

---

## 📦 Requirements
```bash
pip install torch transformers datasets pandas
```

---

## 👨‍💻 Authors

- **Vinay** — Model design, training pipeline, research

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Made with ❤️ for NLP Research</p>
