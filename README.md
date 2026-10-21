# Sparse-CPT

**Sparse-CPT: Cross-Prompt Transferable Test Case Generation for Large Vision-Language Models via Sparse Perturbations**

This repository contains the implementation of **Sparse-CPT**, an automated robustness testing approach for large vision-language models (LVLMs) that generates cross-prompt transferable test cases using sparse visual perturbations.

## Overview

Sparse-CPT extends the cross-prompt adversarial transferability setting of [CroPA](https://github.com/Haochen-Luo/CroPA) by incorporating three complementary mechanisms:

- **Adversarial Vulnerability-Guided Perturbation Localization (AVL):** Identifies image locations with high estimated attack potential and constructs a sparse perturbation mask.
- **Prompt-Aware Gradient Consensus (PGC):** Mitigates conflicts among prompt-specific gradients and modulates perturbation updates according to directional agreement.
- **Cross-Iteration Weighted Perturbation Smoothing (CPS):** Aggregates perturbation states across iterations using an exponential moving average.

## Requirements

Environment configuration and dependency information for Sparse-CPT will be provided with the source code.

## Main Experiments

This repository will provide the code for the main experiments of Sparse-CPT, including test case generation and evaluation of cross-prompt transferability and visual fidelity on three LVLMs:

- OpenFlamingo-9B
- InstructBLIP Vicuna-7B
- BLIP-2 OPT-2.7B


## Acknowledgements

This work builds upon the cross-prompt adversarial transferability framework introduced by [CroPA](https://github.com/Haochen-Luo/CroPA). We sincerely thank the authors of CroPA for making their code and evaluation protocol publicly available. Sparse-CPT extends this research setting by introducing adversarial vulnerability-guided perturbation localization, prompt-aware gradient consensus, and cross-iteration weighted perturbation smoothing.
