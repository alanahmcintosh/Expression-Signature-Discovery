# Expression Signature Discovery

**Benchmarking methods for discovering gene expression signatures associated with genomic alterations**

This repository contains a simulation and benchmarking framework for evaluating methods that infer **gene expression signatures associated with genomic alterations** (mutations, copy number alterations, and gene fusions).

The framework generates realistic simulated datasets based on real cancer transcriptomic data and evaluates how well different statistical and machine learning approaches recover the **true underlying target genes** driving expression changes.

The primary goal is to understand **which computational approaches best identify expression signatures when alterations co-occur and confounding is present**.

---

# Project Overview

Many genomic alterations drive downstream transcriptional programs. Identifying these **gene expression signatures** can help:

- Understand cancer biology
- Identify pathway activation
- Detect mutation-like phenotypes in mutation-negative samples
- Guide targeted therapy discovery

However, recovering these signatures is challenging because:

- Alterations frequently **co-occur**
- Signatures may be **small or weak**
- Sample sizes vary
- Many methods are sensitive to **confounding**

This repository benchmarks multiple approaches to determine **which methods most accurately recover the true target genes underlying simulated alterations**.

---

# Key Features

## Simulation framework

The repository includes a flexible simulation framework that:

- Simulates RNA-seq expression using a **negative binomial distribution**
- Uses real cancer datasets as a reference background
- Injects realistic expression effects associated with genomic alterations
- Models mutation, copy number, and fusion events
- Supports signature sizes from 
- Supports effect sizes ranging from *
- Allows realistic **alteration co-occurrence structures**

## Evaluation pipeline

Predicted signatures are compared against the simulation ground truth using:

- F1 score
- Precision
- Recall
- Accuracy
- Jaccard index
- Matthews correlation coefficient (MCC)

## Robustness analysis

The framework also measures:

- Signal-to-noise ratio (SNR)
- Effective SNR adjusted for confounding
- Alteration co-occurrence statistics
- Signature cohesion and overlap

## Benchmarking framework

Multiple computational methods are benchmarked to assess their ability to recover true gene signatures.

---

# Repository Structure
src/
└── benchmark_sigs/
├── preprocess/
│ Data preprocessing utilities
│
├── simulate/
│ Simulation framework for alterations and RNA expression
│
├── methods/
│ Implementation of supervised and unsupervised signature discovery methods
│
└── evaluation/
│
└──Signature evaluation and benchmarking metrics


### Module descriptions

**preprocess**

Utilities for cleaning and preparing input datasets including:

- mutation preprocessing
- CNA filtering
- RNA preprocessing

**simulate**

Generates synthetic datasets that mimic real cancer transcriptomic data. Components include:

- alteration simulation
- RNA background simulation
- signature generation
- expression effect injection

**methods**

Implements the computational approaches used to derive gene expression signatures.

Current methods include:

- DESeq2
- Lasso
- Elastic Net
- Random Forest
- Support Vector Machine
- Logistic Regression
- Deconfounder
- K-means
- K-means + NMF

**evaluation**

Compares predicted signatures with true simulated signatures and computes benchmarking metrics.

---

# Data Inputs

Each dataset directory contains the following files:

DATASET/
├── alterations_DATASET.csv
├── rna_simulated_DATASET.csv
├── rna_real_DATASET.csv
├── true_signatures_DATASET.json
└── combined_signatures_DATASET.joblib


| File | Description |
|-----|-------------|
| `alterations_*.csv` | Simulated alteration matrix |
| `alt_real_*.csv` | Real alteration matrix used to parameterize simulation |
| `rna_simulated_*.csv` | Simulated RNA expression |
| `rna_real_*.csv` | Real RNA expression used to parameterize simulation |
| `true_signatures_*.json` | Ground truth gene signatures |
| `combined_signatures_*.joblib` | Predicted signatures from all methods |

---

# Running the Benchmark

Typical workflow:

1. Preprocess real datasets
2. Simulate alteration matrices
3. Simulate RNA expression
4. Inject alteration-specific signatures
5. Run signature discovery methods
6. Evaluate predicted signatures against truth

