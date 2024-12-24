# Indigo_Case_Study

# README for Quora Question Answering Project

### [Colab Notebook](https://colab.research.google.com/drive/1VFDUVFEuD9dmFplTub_spkTwfhBC22FY?sharingaction=ownershiptransfer#scrollTo=h7GGwJ4-iQ9v)
### [Presentation](https://docs.google.com/presentation/d/19I5D2c5caEYBTmX_bCxPkYVuMexXMyLng_hz-P8YvNM/edit#slide=id.g2f329bcef73_0_0)

## Overview

This project involves developing a state-of-the-art question-answering model using the Quora Question Answer Dataset. The primary objective was to build an AI system capable of understanding and generating accurate responses to a variety of user queries, mimicking human-like interaction.

---

## Problem Statement

Develop a question-answering model leveraging the Quora Question Answer Dataset to:

- Generate accurate and human-like answers to user queries.
- Evaluate the performance using advanced NLP metrics like ROUGE, BLEU, and F1-score.

---

## Dataset

- **Name**: Quora Question Answer Dataset
- **Source**: [Hugging Face Dataset Repository](https://huggingface.co/datasets/toughdata/quora-question-answer-dataset)
- **Structure**: Contains pairs of `questions` and `answers`.
- **Size**: A reduced dataset of 10,000 records was used due to computational constraints.

---

## Tech Stack

### Backend

- **Python**: Core programming language.
- **Libraries**:
  - **Transformers**: For implementing state-of-the-art NLP models (Hugging Face).
  - **PyTorch**: For model training and evaluation.
  - **NLTK**: For text preprocessing (e.g., tokenization, stopword removal, and lemmatization).
  - **Pandas**: For data manipulation and analysis.
  - **Scikit-learn**: For data splitting.
  - **Evaluate**: For calculating evaluation metrics like ROUGE and BLEU.
  - **Matplotlib & Seaborn**: For visualizations.

### Frontend

- **Jupyter Notebook/Google Colab**: For running the code and displaying results interactively.

### Additional Tools

- **GitHub**: For version control and sharing code.
- **Hugging Face Hub**: For dataset and model repository access.
- **CUDA (GPU)**: For accelerating model training.

---

## Methodology

### 1. Data Exploration, Cleaning, and Preprocessing

- Analyzed the dataset structure for missing values and duplicates.
- Removed duplicates and irrelevant data.
- Applied text preprocessing techniques:
  - **Tokenization**: Splitting text into tokens.
  - **Stopword Removal**: Removed commonly occurring but less meaningful words.
  - **Lemmatization**: Converted words to their base form.

### 2. Model Selection and Training

- **Model**: GPT-2 (from Hugging Face Transformers library).
- **Tokenization**: Used GPT-2 tokenizer with padding and truncation.
- **Datasets**:
  - Training: 80% of the cleaned data.
  - Validation: 20% of the cleaned data.
- **Training Parameters**:
  - Epochs: 5
  - Batch size: 8 (training), 16 (validation)
  - Warmup steps: 500
  - Weight decay: 0.01
  - Evaluation strategy: Epoch-wise

### 3. Evaluation Metrics

- **ROUGE**: Evaluated the overlap of n-grams, word sequences, and word pairs between generated and reference answers.
- **BLEU**: Measured the similarity of generated answers to reference answers.
- **Training Loss and Validation Loss**: Monitored during training.

### 4. Visualization

- Loss vs Epoch: Plotted training and validation loss trends.
- Distribution of ROUGE and BLEU scores: Displayed using histograms.

### 5. Insights and Recommendations

- **Model Insights**:
  - Validation loss stabilized after 3 epochs, indicating convergence.
  - ROUGE and BLEU scores showed high overlap and similarity, reflecting the model's ability to generate accurate answers.
- **Recommendations**:
  - Experiment with larger models like GPT-3 or T5 for improved accuracy.
  - Fine-tune on specific domains for specialized question-answering tasks.
  - Use more diverse datasets to handle broader query types.

---

## Results

- **Training Loss**: Decreased consistently across epochs, indicating effective learning.
- **Validation Loss**: Plateaued after 3 epochs, demonstrating convergence.
- **ROUGE Score**: Mean score of 0.9116.
- **BLEU Score**: Mean score of 0.8108.

### Visualizations

1. **Epoch vs. Loss**:
   - Training loss decreased steadily.
   - Validation loss stabilized after a few epochs.
2. **ROUGE and BLEU Distributions**:
   - Both metrics showed higher scores concentrated near 1.0, reflecting accurate predictions.

---
