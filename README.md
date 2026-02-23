This is a professional README.md tailored for this project. It emphasizes the methodology, resource-constrained focus, and reproducibility.

🏗️ Recommended Repository Structure

Before you upload, ensure your folder structure looks like this:

├── data/                   # Dataset (SST-2)
├── models/                 # Model definitions (HAT-D, Baselines)
├── results/                # Plots, CSVs, and Artifacts
├── checkpoints/            # (Optional) Saved .pth files
├── notebooks/              # Colab/Kaggle .ipynb files
├── src/                    # Python scripts for training/eval
└── README.md

📄 README.md Template
# Lightweight Hybrid Adversarial Training for Resource-Constrained Sentiment Analysis (HAT-D)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

This repository contains the implementation and experimental framework for my Master's dissertation: **"Balancing Robustness, Accuracy, and Efficiency in NLP via Hybrid Adversarial Training."**

The project introduces **HAT-D**, a lightweight defense mechanism designed to protect Transformer-based sentiment classifiers (DistilBERT/TinyBERT) against adversarial attacks in resource-constrained environments.

## 🚀 Key Features
- **Hybrid Defense:** Combines TRADES-based adversarial training with Randomized Smoothing.
- **Multi-Attack Evaluation:** Robustness testing against Synonym Substitution, Character-level Edits, and Mixed-Strategy attacks.
- **Efficiency-First:** Optimized for low-latency inference and minimal GPU memory footprint.
- **Ablation Framework:** Systematic testing of individual defense components (Adversarial Training vs. Smoothing).

## 📊 Methodology
The framework utilizes a "Denoising" approach where adversarial perturbations are mitigated at the embedding level. 
- **Base Model:** DistilBERT-base-uncased.
- **Dataset:** SST-2 (Stanford Sentiment Treebank).
- **Optimization:** Mixed Precision (AMP), Gradient Accumulation, and Cosine Annealing.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kudzo90/adversarial-nlp-hatd.git
   cd adversarial-nlp-hatd

Install dependencies:
pip install -r requirements.txt

📈 Usage
Training & Ablation Study

To run the full ablation matrix (Vanilla, Adv-Only, Smooth-Only, HAT-D):

python run_ablation.py --model distilbert --epochs 5 --batch_size 32

Evaluation

To evaluate a saved checkpoint against adversarial attacks:

python evaluate_robustness.py --checkpoint checkpoints/HATD_DistilBERT.pth

🧪 Results (Preliminary)
Model	Clean Acc	Mixed Attack Acc	Latency (ms)
Vanilla DistilBERT	94.3%	84.8%	83.4
HAT-D (Ours)	93.7%	89.2%	89.8

Note: HAT-D achieves significantly higher robustness with a negligible (<1%) drop in clean accuracy.

📂 Artifacts Generated

The pipeline automatically generates the following for dissertation analysis:

pareto_frontier.png: Visualizing the Accuracy-Robustness trade-off.
attack_success_rates.csv: Detailed breakdown of model vulnerability.
training_curves.png: Loss and Accuracy convergence plots.
🎓 Dissertation Context

This research addresses the gap in deploying robust NLP models on edge devices. By focusing on lightweight architectures and hybrid training objectives, we demonstrate that adversarial security does not necessitate prohibitive computational overhead.

📜 License

Distributed under the MIT License. See LICENSE for more information.

✉️ Contact

Wonder Ekpe Project Link: https://github.com/Kudzo90/adversarial-nlp-hatd


***
