# Sparse-CPT

**Sparse-CPT: Cross-Prompt Transferable Test Case Generation for Large Vision-Language Models via Sparse Perturbations**

Official implementation of **Sparse-CPT**, an automated robustness testing approach for large vision-language models (LVLMs).

Sparse-CPT generates cross-prompt transferable test cases using sparse visual perturbations, aiming to preserve visual fidelity while maintaining effective cross-prompt transferability.

## Overview

Sparse-CPT consists of three complementary components:

- **Adversarial Vulnerability-Guided Perturbation Localization (AVL):** Identifies image locations with higher estimated attack potential and constructs a sparse perturbation mask.
- **Prompt-Aware Gradient Consensus (PGC):** Mitigates gradient conflicts across different prompts and modulates perturbation updates based on directional agreement.
- **Cross-Iteration Weighted Perturbation Smoothing (CPS):** Aggregates perturbation states across iterations using an exponential moving average.

## Evaluated Models

Our experiments cover three LVLMs:

- OpenFlamingo-9B
- InstructBLIP Vicuna-7B
- BLIP-2 OPT-2.7B

We evaluate cross-prompt transferability and visual fidelity across six predefined target texts and four downstream tasks.

## Installation and Usage

Installation instructions, model preparation, and reproduction commands will be provided with the corresponding source code.

## Experimental Results

The repository will provide materials for reproducing the main experimental results, component ablation studies, and hyperparameter sensitivity analyses.

## Citation

Citation information will be updated upon publication.
