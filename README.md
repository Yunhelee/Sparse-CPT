# Sparse-CPT

**Sparse-CPT: Cross-Prompt Transferable Test Case Generation for Large Vision-Language Models via Sparse Perturbations**

This repository contains the implementation of **Sparse-CPT**, an automated robustness testing approach for large vision-language models (LVLMs) that generates cross-prompt transferable test cases using sparse visual perturbations.

## Overview

Sparse-CPT extends the cross-prompt adversarial transferability setting of [CroPA](https://github.com/Haochen-Luo/CroPA) by incorporating three complementary mechanisms:

- **Adversarial Vulnerability-Guided Perturbation Localization (AVL):** Identifies image locations with high estimated attack potential and constructs a sparse perturbation mask.
- **Prompt-Aware Gradient Consensus (PGC):** Mitigates conflicts among prompt-specific gradients and modulates perturbation updates according to directional agreement.
- **Cross-Iteration Weighted Perturbation Smoothing (CPS):** Aggregates perturbation states across iterations using an exponential moving average.

## Release scope

The two confirmed entry points are:

- `test.py`: Generates test cases using a trained perturbation generator.
- `train.py`: Trains the perturbation generator.

## Environment

The recorded server environment uses Python 3.9, PyTorch 2.0.1 with CUDA 11.7, and torchvision 0.15.2. The project uses CroPA's customized Transformers implementation based on version 4.33.2; installing the stock Transformers package alone is not an equivalent setup.

## Requirements

Environment configuration and dependency information for Sparse-CPT will be provided with the source code.

## Main Experiments

This repository provide the code for the main experiments of Sparse-CPT, including test case generation and evaluation of cross-prompt transferability and visual fidelity on three LVLMs:

- OpenFlamingo-9B
- InstructBLIP Vicuna-7B
- BLIP-2 OPT-2.7B

## Usage

### Train the Perturbation Generator

```bash
python pretrained_model/train.py \
  --model_name open_flamingo \
  --target_text "unknown" \
  --clip_model_path /path/to/clip-vit-large-patch14 \
  --save_dir /path/to/save/checkpoints
```

### Generate Test Cases

```bash
python test.py \
  --model_name open_flamingo \
  --target_text "unknown" \
  --generator_checkpoint /path/to/generator.pt \
  --clip_model_path /path/to/clip-vit-large-patch14 \
  --use_generator_sparse_mask \
  --prompt_grad_method pcgrad_graddrop \
```


## Acknowledgements

This work builds upon the cross-prompt adversarial transferability framework introduced by [CroPA](https://github.com/Haochen-Luo/CroPA). We sincerely thank the authors of CroPA for making their code and evaluation protocol publicly available. Sparse-CPT extends this research setting by introducing adversarial vulnerability-guided perturbation localization, prompt-aware gradient consensus, and cross-iteration weighted perturbation smoothing.
