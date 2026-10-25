# Statistical Methods in AI — Assignment 4

This repository contains the notebooks and trained models for Assignment 4
(MLP, IOC-NN and Clustering).

## Links

### Weights & Biases

Public project:

- [SMAI A4 MLP — WandB project](https://forge.coreweave.com/wandb/suryamanojphy31-iiit-hyderabad/smai-a4-mlp?nw=nwusersuryamanojphy31)

### Hugging Face models

- MLP: [Surya4/smai-a4-mlp-p0](https://huggingface.co/Surya4/smai-a4-mlp-p0)
- IOC-NN: [Surya4/smai-a4-iocnn-p0](https://huggingface.co/Surya4/smai-a4-iocnn-p0)

These are the two clean-label (`p = 0`) models compared in Section 1.5. The
architecture was selected in Section 1.2.5; the uploaded MLP is the clean-label
PyTorch model used in the 1.5 comparison.

## Notebooks

- `mlp.ipynb` — Question 1: MLP from scratch, IOC-NN and label-noise experiment
- `clustering.ipynb` — Question 2: GMM, agglomerative clustering, DBSCAN and mask refinement

## Models

Both models use the architecture selected in Section 1.2.5:

```text
3072 → 128 → 10
```

- Parameters: `394,634`
- Loss: categorical cross-entropy
- Optimizer: SGD
- Learning rate: `0.02`
- Batch size: `64`
- Epochs: `100`
- No early stopping in the 1.4 label-noise experiment

### MLP

- Hidden activation: ReLU
- Output: 10 logits, Identity activation
- Clean-label test accuracy: `39.67%`
- Clean-label test loss: `5.0191`

### IOC-NN

- Hidden activation: ELU
- Output: 10 logits, Identity activation
- Layers 2 and later have non-negative weights; after every optimizer step,
  negative weights are replaced by `exp(w - 5)`
- Clean-label test accuracy: `40.24%`
- Clean-label test loss: `3.9075`

Each Hugging Face repository contains:

- `model.pt` — PyTorch `state_dict`
- `config.json` — architecture and training details
- `preprocessing.npz` — CIFAR-10 train mean/std arrays
- `README.md` — model card

## Preprocessing

CIFAR-10 images are:

1. read as RGB `uint8` images of shape `32 × 32 × 3`;
2. flattened row-major to `3072` values;
3. divided by `255`;
4. standardised per column with the mean and standard deviation of the
   CIFAR-10 training split;
5. cast to `float32` for PyTorch.

The mean and standard deviation arrays are saved in `preprocessing.npz`
(keys: `mean`, `std`).

## Reproducibility

- Q1 seed: `1036` (`roll_number % 100000`)
- PyTorch model initialisation seed: `SEED + 1`
- DataLoader generator seed: `SEED + 2`
- Q1 training runs are logged in the public WandB project linked above.
- Both notebooks are intended to run from a fresh kernel.
