# 🎯 CIFAR-10 Experiments Guide

Same experimental pipeline as MNIST, but on the more challenging CIFAR-10 dataset!

---

## 🚀 Quick Start

### Test Run (10 epochs, fast)
```bash
python experiments/cifar10_mcmc_experiment.py \
    --loss-type ce \
    --epochs 10 \
    --batch-size 128 \
    --plot-dir experiments/plots
```

### Full Training (160 epochs, ~2-3 hours on GPU)
```bash
python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_annealing \
    --beta-start 0.5 \
    --beta-end 5.0 \
    --b-value 5.0 \
    --epochs 160 \
    --batch-size 128 \
    --plot-dir experiments/plots
```

---

## 📊 All Available Loss Functions

Same as MNIST experiments:

| Loss Type | Command | Annealing |
|-----------|---------|-----------|
| Cross-Entropy | `--loss-type ce` | None |
| Hinge | `--loss-type hinge` | None |
| Hinge + β-Annealing | `--loss-type hinge_beta_annealing` | β: 0.5→5.0 |
| Hinge + b-Annealing | `--loss-type hinge_b_annealing` | b: 1.0→100.0 |
| Hinge + Both | `--loss-type hinge_both_annealing` | Both |
| Vlog Fixed | `--loss-type vlog_fixed` | None |
| Vlog + β-Annealing | `--loss-type vlog_annealing` | β: 0.5→5.0 |
| Vlog + b-Annealing | `--loss-type vlog_b_annealing` | b: 1.0→100.0 |
| Vlog + Both | `--loss-type vlog_both_annealing` | Both |

---

## 🎓 Key Differences from MNIST

### Dataset
- **Size**: 50,000 train + 10,000 test images
- **Resolution**: 32×32×3 (RGB) vs 28×28×1 (grayscale)
- **Difficulty**: Much harder than MNIST!

### Model
- **Architecture**: VGG BinaryNet (conv layers) vs MLP BinaryNet
- **Parameters**: More complex network for images

### Training
- **Default epochs**: 160 (vs 10-20 for MNIST)
- **Learning rate schedule**: Decreases at epochs 40, 80, 100
- **Batch size**: 128 (vs 4096 for MNIST) - larger batches may cause memory issues
- **Training time**: ~1 hour for 10 epochs on GPU

### Expected Accuracy
- **Cross-Entropy baseline**: ~85-90%
- **Vlog + β-annealing**: Target 85-92%
- Much lower than MNIST (97%+) because CIFAR-10 is harder!

---

## 🔬 Recommended Experiment Sequence

### Phase 1: Quick Baseline (10 epochs each)
Test if setup works and get rough accuracy estimates:

```bash
# Baseline
python experiments/cifar10_mcmc_experiment.py --loss-type ce --epochs 10 --batch-size 128

# Best from MNIST
python experiments/cifar10_mcmc_experiment.py --loss-type vlog_annealing --b-value 5.0 --beta-start 0.5 --beta-end 5.0 --epochs 10

# Reverse b-annealing (promising from MNIST)
python experiments/cifar10_mcmc_experiment.py --loss-type vlog_b_annealing --vlog-b-start 10.0 --vlog-b-end 1.0 --epochs 10
```

### Phase 2: Selected Full Training (160 epochs)
Run full training for the most promising configurations:

```bash
# If quick tests look good, run full training
python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_annealing \
    --b-value 5.0 \
    --beta-start 0.5 \
    --beta-end 5.0 \
    --epochs 160 \
    --batch-size 128
```

---

## 💡 Hyperparameter Recommendations

Based on MNIST findings:

### ✅ Likely to Work Well
- `--b-value 1.0` to `5.0` for fixed Vlog
- β-annealing: `--beta-start 0.5 --beta-end 5.0`
- Reverse b-annealing: `--vlog-b-start 10.0 --vlog-b-end 1.0`

### ⚠️ May Need Adjustment
- Forward b-annealing (1→100): Likely unstable, but worth trying
- Both annealing: Unknown for CIFAR-10

### ❌ Avoid
- Very high b values (b=20.0+): Unstable on MNIST
- Hinge + b-annealing: Failed catastrophically on MNIST
- Extremely aggressive annealing schedules

---

## 🎯 Memory and Speed Tips

### GPU Memory Issues?
```bash
# Reduce batch size
--batch-size 64  # or even 32

# Or use gradient accumulation (would need to implement)
```

### Speed Up Training?
```bash
# Use more workers
--num-workers 8

# Test with fewer epochs first
--epochs 10

# Use larger batch size if memory allows
--batch-size 256
```

### Monitor Training?
```bash
# More frequent logging
--log-interval 50  # default is 100

# Less frequent logging (faster)
--log-interval 200
```

---

## 📁 Output Structure

All outputs go to same structure as MNIST:

```
experiments/
├── plots/
│   ├── cifar10_ce_e10_bs128_lr0.005.png
│   ├── cifar10_vlog_annealing_b5.0_beta0.5-5.0_e10_bs128_lr0.005.png
│   └── ...
└── results/
    ├── cifar10_ce_e10_bs128_lr0.005.txt
    ├── cifar10_vlog_annealing_b5.0_beta0.5-5.0_e10_bs128_lr0.005.txt
    └── ...
```

---

## 🔍 Analyzing Results

Use the same analysis script - it will automatically detect CIFAR-10 results:

```bash
cd experiments/summary_results
python analyze_all_results.py
```

The analysis will show:
- MNIST results (mnist_*)
- CIFAR-10 results (cifar10_*)
- Comparison tables for each dataset

---

## 📊 Example Commands for Key Experiments

### 1. Baseline Comparison
```bash
# Cross-Entropy baseline
python experiments/cifar10_mcmc_experiment.py --loss-type ce --epochs 160

# Vlog Fixed (best b from MNIST)
python experiments/cifar10_mcmc_experiment.py --loss-type vlog_fixed --b-value 2.0 --epochs 160
```

### 2. β-Annealing
```bash
# Vlog + β-annealing
python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_annealing \
    --b-value 5.0 \
    --beta-start 0.5 \
    --beta-end 5.0 \
    --epochs 160
```

### 3. Reverse b-Annealing (Novel!)
```bash
# Reverse: easier optimization landscape over time
python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_b_annealing \
    --vlog-b-start 10.0 \
    --vlog-b-end 1.0 \
    --epochs 160
```

### 4. Combined (if stable)
```bash
# Both annealing strategies
python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_both_annealing \
    --vlog-b-start 10.0 \
    --vlog-b-end 1.0 \
    --beta-start 0.5 \
    --beta-end 5.0 \
    --epochs 160
```

---

## ⏱️ Time Estimates (V100 GPU)

| Epochs | Batch Size | Approximate Time |
|--------|-----------|------------------|
| 10 | 128 | ~1 hour |
| 50 | 128 | ~4 hours |
| 160 | 128 | ~12-15 hours |
| 10 | 256 | ~40 min |

**Pro tip**: Run 10-epoch tests first to verify everything works before committing to 160 epochs!

---

## 🎓 Research Questions for CIFAR-10

1. **Do MNIST findings transfer?**
   - Does Vlog + β-annealing still outperform CE?
   - Is reverse b-annealing still stable?

2. **Are effects stronger on harder datasets?**
   - CIFAR-10 is harder → margin-based losses might help more

3. **What's the accuracy gap?**
   - Compare MNIST improvement (+1.95%) to CIFAR-10 improvement

---

**Good luck with CIFAR-10 experiments! 🚀**

