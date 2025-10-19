# CIFAR-10 Experiment Commands
# Generated from MNIST plot analysis

## Overview
This file contains CIFAR-10 commands matching the hyperparameters from MNIST experiments.

**Key Adjustments for CIFAR-10:**
- Batch size: 512 → 128 (CIFAR-10 images are larger)
- Learning rate: 0.01 → 0.005 (recommended for CIFAR-10)
- Epochs: Same as MNIST for direct comparison
- All loss hyperparameters: **Identical to MNIST**

---

## 📋 Experiment Summary (15 Total)

| # | Loss Type | MNIST Params | CIFAR-10 Params |
|---|-----------|--------------|-----------------|
| 1 | Cross-Entropy | e10, bs512 | e10, bs128 |
| 2 | Hinge | m1.0, e10, bs512 | m1.0, e10, bs128 |
| 3 | Hinge b-annealing | b:1→100, e10, bs512 | b:1→100, e10, bs128 |
| 4 | Hinge β-annealing | β:0.5→5.0, e10, bs512 | β:0.5→5.0, e10, bs128 |
| 5 | Hinge BOTH | b:1→100, β:0.5→5.0, e10 | b:1→100, β:0.5→5.0, e10 |
| 6 | Vlog Fixed (b=1.0) | b=1.0, e10, bs512 | b=1.0, e10, bs128 |
| 7 | Vlog Fixed (b=2.0) | b=2.0, e10, bs512 | b=2.0, e10, bs128 |
| 8 | Vlog Fixed (b=5.0) | b=5.0, e10, bs512 | b=5.0, e10, bs128 |
| 9 | Vlog Fixed (b=20.0) | b=20.0, e10, bs512 | b=20.0, e10, bs128 |
| 10 | Vlog β-annealing | b=5.0, β:0.5→5.0, e10 | b=5.0, β:0.5→5.0, e10 |
| 11 | Vlog β-annealing | b=5.0, β:0.5→100.0, e10 | b=5.0, β:0.5→100.0, e10 |
| 12 | Vlog β-annealing | b=5.0, β:0.5→100.0, e20 | b=5.0, β:0.5→100.0, e20 |
| 13 | Vlog b-annealing | b:1→10, e10, bs512 | b:1→10, e10, bs128 |
| 14 | Vlog b-annealing | b:1→100, e10, bs512 | b:1→100, e10, bs128 |
| 15 | Vlog b-annealing (REVERSE!) | b:10→1, e10, bs512 | b:10→1, e10, bs128 |

---

## 🚀 Commands for Google Colab

### **Setup (run once)**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone or navigate to repo
%cd /content/drive/MyDrive/path/to/THESIS

# Set plot directory
plot_dir = "BinaryNet.pytorch/experiments/plots"
```

---

## 1️⃣ **Cross-Entropy Baseline**

```python
# MNIST: mnist_ce_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type ce \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 2️⃣ **Hinge Loss (Standard)**

```python
# MNIST: mnist_hinge_m1.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type hinge \
    --hinge-margin 1.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 3️⃣ **Hinge + b-Annealing**

```python
# MNIST: mnist_hinge_b_annealing_m1.0_b1.0-100.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type hinge_b_annealing \
    --hinge-margin 1.0 \
    --hinge-b-start 1.0 \
    --hinge-b-end 100.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 4️⃣ **Hinge + β-Annealing**

```python
# MNIST: mnist_hinge_beta_annealing_m1.0_beta0.5-5.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type hinge_beta_annealing \
    --hinge-margin 1.0 \
    --hinge-beta-start 0.5 \
    --hinge-beta-end 5.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 5️⃣ **Hinge + BOTH Annealing**

```python
# MNIST: mnist_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type hinge_both_annealing \
    --hinge-margin 1.0 \
    --hinge-b-start 1.0 \
    --hinge-b-end 100.0 \
    --hinge-beta-start 0.5 \
    --hinge-beta-end 5.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 6️⃣ **Vlog Fixed (b=1.0)**

```python
# MNIST: mnist_vlog_fixed_b1.0_beta1.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_fixed \
    --b-value 1.0 \
    --beta-fixed 1.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 7️⃣ **Vlog Fixed (b=2.0)**

```python
# MNIST: mnist_vlog_fixed_b2.0_beta1.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_fixed \
    --b-value 2.0 \
    --beta-fixed 1.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 8️⃣ **Vlog Fixed (b=5.0)**

```python
# MNIST: mnist_vlog_fixed_b5.0_beta1.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_fixed \
    --b-value 5.0 \
    --beta-fixed 1.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 9️⃣ **Vlog Fixed (b=20.0)**

```python
# MNIST: mnist_vlog_fixed_b20.0_beta1.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_fixed \
    --b-value 20.0 \
    --beta-fixed 1.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 🔟 **Vlog + β-Annealing (β: 0.5→5.0)**

```python
# MNIST: mnist_vlog_annealing_b5.0_beta0.5-5.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_annealing \
    --b-value 5.0 \
    --beta-start 0.5 \
    --beta-end 5.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 1️⃣1️⃣ **Vlog + β-Annealing (β: 0.5→100.0)**

```python
# MNIST: mnist_vlog_annealing_b5.0_beta0.5-100.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_annealing \
    --b-value 5.0 \
    --beta-start 0.5 \
    --beta-end 100.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 1️⃣2️⃣ **Vlog + β-Annealing (β: 0.5→100.0, 20 epochs)**

```python
# MNIST: mnist_vlog_annealing_b5.0_beta0.5-100.0_e20_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_annealing \
    --b-value 5.0 \
    --beta-start 0.5 \
    --beta-end 100.0 \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 1️⃣3️⃣ **Vlog + b-Annealing (b: 1→10)**

```python
# MNIST: mnist_vlog_b_annealing_b1.0-10.0_beta1.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_b_annealing \
    --vlog-b-start 1.0 \
    --vlog-b-end 10.0 \
    --beta-fixed 1.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 1️⃣4️⃣ **Vlog + b-Annealing (b: 1→100)**

```python
# MNIST: mnist_vlog_b_annealing_b1.0-100.0_beta1.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_b_annealing \
    --vlog-b-start 1.0 \
    --vlog-b-end 100.0 \
    --beta-fixed 1.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 1️⃣5️⃣ **Vlog + b-Annealing REVERSE (b: 10→1)** ⭐ **Novel!**

```python
# MNIST: mnist_vlog_b_annealing_b10.0-1.0_beta1.0_e10_bs512_lr0.01.png
!python experiments/cifar10_mcmc_experiment.py \
    --loss-type vlog_b_annealing \
    --vlog-b-start 10.0 \
    --vlog-b-end 1.0 \
    --beta-fixed 1.0 \
    --epochs 10 \
    --batch-size 128 \
    --lr 0.005 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

---

## 📊 After Running All Experiments

Generate the summary report:

```python
%cd BinaryNet.pytorch/experiments/summary_results
!python analyze_all_results.py
```

This will create:
- `experiment_summary_report.html` (with embedded plots)
- `experiment_summary_report.md` (GitHub-compatible)

---

## ⏱️ Time Estimates

| Epochs | Estimated Time per Experiment | Total for 15 Experiments |
|--------|-------------------------------|--------------------------|
| 10 | ~20 min (with GPU) | ~5 hours |
| 20 | ~40 min (with GPU) | ~10 hours |

**Recommendation:** Run overnight on Colab with GPU enabled.

---

## 🔄 Batch Execution Script

To run all 15 experiments sequentially:

```python
# List of all commands
commands = [
    # 1. CE
    'python experiments/cifar10_mcmc_experiment.py --loss-type ce --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    
    # 2. Hinge
    'python experiments/cifar10_mcmc_experiment.py --loss-type hinge --hinge-margin 1.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    
    # 3. Hinge b-annealing
    'python experiments/cifar10_mcmc_experiment.py --loss-type hinge_b_annealing --hinge-margin 1.0 --hinge-b-start 1.0 --hinge-b-end 100.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    
    # 4. Hinge beta-annealing
    'python experiments/cifar10_mcmc_experiment.py --loss-type hinge_beta_annealing --hinge-margin 1.0 --hinge-beta-start 0.5 --hinge-beta-end 5.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    
    # 5. Hinge both
    'python experiments/cifar10_mcmc_experiment.py --loss-type hinge_both_annealing --hinge-margin 1.0 --hinge-b-start 1.0 --hinge-b-end 100.0 --hinge-beta-start 0.5 --hinge-beta-end 5.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    
    # 6-9. Vlog Fixed (different b values)
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_fixed --b-value 1.0 --beta-fixed 1.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_fixed --b-value 2.0 --beta-fixed 1.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_fixed --b-value 5.0 --beta-fixed 1.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_fixed --b-value 20.0 --beta-fixed 1.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    
    # 10-12. Vlog beta-annealing
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_annealing --b-value 5.0 --beta-start 0.5 --beta-end 5.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_annealing --b-value 5.0 --beta-start 0.5 --beta-end 100.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_annealing --b-value 5.0 --beta-start 0.5 --beta-end 100.0 --epochs 20 --batch-size 128 --lr 0.005 --num-workers 4',
    
    # 13-15. Vlog b-annealing
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_b_annealing --vlog-b-start 1.0 --vlog-b-end 10.0 --beta-fixed 1.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_b_annealing --vlog-b-start 1.0 --vlog-b-end 100.0 --beta-fixed 1.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
    'python experiments/cifar10_mcmc_experiment.py --loss-type vlog_b_annealing --vlog-b-start 10.0 --vlog-b-end 1.0 --beta-fixed 1.0 --epochs 10 --batch-size 128 --lr 0.005 --num-workers 4',
]

# Run all experiments
import subprocess
import time

for i, cmd in enumerate(commands, 1):
    print(f"\n{'='*70}")
    print(f"Starting Experiment {i}/15")
    print(f"{'='*70}\n")
    start_time = time.time()
    
    subprocess.run(cmd, shell=True)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Experiment {i}/15 completed in {elapsed/60:.1f} minutes")
    print(f"Remaining: {15-i} experiments")

print("\n🎉 All 15 experiments completed!")
```

---

## 📝 Notes

1. **Batch size:** Reduced from 512 (MNIST) to 128 (CIFAR-10) due to larger images
2. **Learning rate:** Reduced from 0.01 to 0.005 (standard for CIFAR-10)
3. **All other hyperparameters:** Kept identical to MNIST for fair comparison
4. **P0 bugs:** Already fixed! This run will use correct code.

---

## 🎯 Priority Experiments

If you want to run a subset first:

**Quick Test (3 experiments, ~1 hour):**
1. Cross-Entropy baseline (#1)
2. Vlog + β-annealing (#10) - Best from MNIST
3. Vlog + b-annealing REVERSE (#15) - Novel discovery

**Core Comparison (6 experiments, ~2 hours):**
Add: Hinge baseline (#2), Vlog fixed b=5.0 (#8), Vlog b-annealing forward (#14)

