# Fake News Detection Pipeline

**BERT 81.2%** vs baselines on PolitiFact + GossipCop (balanced datasets).[1]

## 📊 Results

| Model             | Accuracy | F1-Score |
|-------------------|----------|----------|
| **BERT**          | **81.2%**| **80.9%**|
| Logistic Regression| 79.2%   | 78.7%   |
| Naive Bayes       | 78.2%   | 78.0%   |
| BiLSTM            | 77.2%   | 77.4%   |[1]

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train_complete_pipeline.py  # ~30min on RTX GPU
```

## 🗂️ Datasets

Place CSVs in `D:\fakeNewsPoli\` (or update paths):

```
D:\fakeNewsPoli\
├── politifact_real.csv
├── politifact_fake.csv
├── gossipcop_real.csv
└── gossipcop_fake.csv
```

**Column**: `title` (text data)

## 📁 Outputs

```
✅ logistic_regression_model.pkl (79.2%)
✅ naive_bayes_model.pkl (78.2%) 
✅ bilstm_model_cross_domain.pt (77.2%)
✅ bert_model_cross_domain.pt (81.2% - BEST)
✅ tfidf_vectorizer.pkl
✅ bilstm_vocab.pkl
✅ training_results_comparison.csv
```

## ⚙️ Requirements

```txt
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

## ✨ Features

- **4 Models**: BERT, BiLSTM, Logistic, Naive Bayes
- **GPU Ready**: Auto CUDA + mixed precision (FP16)
- **Cross-Domain**: PolitiFact + GossipCop
- **Balanced**: Auto real/fake balancing
- **Production**: All models + artifacts saved

## 🔧 GPU Tips

- **OOM?** `BATCH_SIZE=16` or `MAX_LENGTH=128`
- **CPU slow?** Install CUDA PyTorch
- **Paths wrong?** Edit top of script

***

**⭐ & 🚀 if BERT beats your baselines!**[1]
