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
