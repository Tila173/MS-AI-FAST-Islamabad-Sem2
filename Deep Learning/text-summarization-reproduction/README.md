# 📄 Reproducing Transformer-Based Text Summarization

## 📖 Overview

This project reproduces the results of the research paper:

**“Deep Learning for Text Summarization Using NLP for Automated News Digest”**

The goal is to evaluate and compare the performance of multiple transformer-based models for abstractive text summarization using a consistent experimental setup.

---

## 📄 Original Paper

* **Title:** Deep Learning for Text Summarization Using NLP for Automated News Digest
* **Authors:** Rani Krishna et al.
* **Journal:** Scientific Reports (2025)
* **DOI:** https://doi.org/10.1038/s41598-025-20224-1

---

## 🤖 Models Used

The following pre-trained transformer models were evaluated:

* T5 (Base & Large)
* BART Large CNN
* PEGASUS Large

All models were fine-tuned on the same dataset using identical preprocessing and comparable hyperparameters.

---

## 📊 Dataset

* **Dataset:** CNN/DailyMail
* ~300,000 news articles with human-written summaries
* Standard split:

  * Train: 287,113
  * Validation: 13,368
  * Test: 11,490

---

## ⚙️ Experimental Setup

* Framework: PyTorch
* Library: Hugging Face Transformers
* Hardware: 4 × NVIDIA Tesla V100 GPUs (32GB each)
* CUDA Version: 12.0

Each model was trained on a dedicated GPU to enable parallel experimentation.

---

## 🔧 Preprocessing

* Tokenization using pretrained tokenizers
* Padding and truncation applied for uniform sequence lengths
* Max input length:

  * T5: 512 tokens
  * BART & PEGASUS: 1024 tokens
* Target summaries limited to 128 tokens
* Padding tokens ignored in loss computation (`-100`)

---

## ⚙️ Hyperparameters

| Parameter     | T5-base | T5-large | BART-large-CNN | PEGASUS-large |
| ------------- | ------- | -------- | -------------- | ------------- |
| Learning Rate | 3e-4    | 3e-4     | 3e-5           | 1e-4          |
| Batch Size    | 64      | 32       | 64             | 64            |
| Epochs        | 3       | 2        | 2              | 2             |
| Optimizer     | AdamW   | AdamW    | AdamW          | AdamW         |
| Warmup Steps  | 2000    | 2000     | 2000           | 2000          |

---

## 📊 Results

### 🔹 After Fine-Tuning

| Model          | ROUGE-1    | ROUGE-2    | ROUGE-L    | BLEU       |
| -------------- | ---------- | ---------- | ---------- | ---------- |
| T5-base        | 0.0000 ❌   | 0.0000     | 0.0000     | 0.0000     |
| T5-large       | 0.4176     | 0.1933     | 0.2939     | 0.0756     |
| BART-large-CNN | **0.4475** | **0.2148** | 0.3079     | **0.1102** |
| PEGASUS-large  | 0.4447     | 0.2138     | **0.3128** | 0.1082     |

---

## 📈 Key Observations

* **BART-large-CNN** achieved the best overall performance
* **PEGASUS-large** performed best in ROUGE-L
* **T5-large** showed stable but limited improvement
* Reproduced results were **higher than the original paper**, likely due to:

  * Updated pretrained checkpoints
  * Better hardware (multi-GPU setup)
  * Implementation differences

---

## ⚠️ T5-base Training Collapse

During fine-tuning, T5-base failed due to numerical instability:

* `grad_norm = NaN`
* Loss collapsed to 0
* Final evaluation scores: 0.0000

### Possible Causes:

* High learning rate (3e-4)
* Mixed precision training (fp16)
* Gradient explosion / instability

### Suggested Fixes:

* Reduce learning rate (e.g., 1e-4 or 5e-5)
* Disable fp16
* Train fewer epochs

---

## 📁 Repository Structure

```
text-summarization-reproduction/
│
├── logs/           # Training logs for each model
├── results/        # Evaluation results
├── references-predictions/
├── tf-idf/        # Extractive Baseline
├── cross-domain-evaluation/        # Cross Domain
└── README.md
```

---

## 📂 Logs

Training logs (loss, gradients, evaluation metrics) are available in the `/logs` directory.

---

## 📄 Report

Full detailed report is available here:

📎 `docs/report.pdf`

---

## ⚠️ Large Files Notice

Pretrained model weights are not included due to size limitations.
(They can be shared via external storage if needed.)

---

## 🔁 Reproducibility

To reproduce the experiments:

```bash
git clone https://github.com/Tila173/text-summarization-reproduction.git
cd text-summarization-reproduction
pip install -r requirements.txt
python train.py
```

---

## 📚 References
1. PEGASUS Paper (Zhang et al., 2020)
2. BART Paper (Lewis et al., 2020)
3. T5 Paper (Raffel et al., 2020)
4. CNN/DailyMail Dataset
---
## 👨‍💻 Author
Tila Muhammad  
BS Artificial Intelligence  
FAST-NUCES Islamabad  
## 🤝 Contributors
- 25i-7601  
- 25i-7639  
- 25i-7641  
---
## ⭐ Acknowledgment

This project was completed as part of an academic assignment focused on reproducibility in deep learning research.
