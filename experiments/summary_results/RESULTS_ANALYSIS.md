# Binary Neural Network - Experimental Results

**Total experiments: 35**

## Table of Contents

- [CIFAR-10](#cifar10)
- [MNIST](#mnist)
- [Training Curves](#training-curves)

---

# CIFAR-10

**20 experiments** (14 successful, 6 failed)

| Filename | Loss | Best Acc | Final Acc | Epochs | BS | LR | Time | Status |
|----------|------|----------|-----------|--------|----|----|------|--------|
| `cifar10_ce_e10_bs512_lr0.0005` | ce | **54.33%** | 53.94% | 10 | 512 | 0.0005 | 17.9min | ✅ |
| `cifar10_ce_e10_bs512_lr0.005` | ce | **57.29%** | 57.29% | 10 | 512 | 0.005 | 16.3min | ✅ |
| `cifar10_ce_e10_bs512_lr0.01` | ce | **57.68%** | 57.42% | 10 | 512 | 0.01 | 17.6min | ✅ |
| `cifar10_ce_e10_bs512_lr0.1` | ce | **45.49%** | 45.24% | 10 | 512 | 0.1 | 17.0min | ❌ |
| `cifar10_ce_e160_bs128_lr0.005` | ce | **67.73%** | 66.64% | 160 | 128 | 0.005 | 153.6min | ✅ |
| `cifar10_ce_e20_bs2056_lr0.01` | ce | **58.79%** | 58.79% | 20 | 2056 | 0.01 | 4.8min | ✅ |
| `cifar10_ce_e20_bs512_lr0.01` | ce | **61.28%** | 61.28% | 20 | 512 | 0.01 | 4.7min | ✅ |
| `cifar10_ce_e50_bs2056_lr0.0001` | ce | **56.99%** | 56.99% | 50 | 2056 | 0.0001 | 11.9min | ✅ |
| `cifar10_ce_e50_bs2056_lr0.001` | ce | **61.10%** | 61.10% | 50 | 2056 | 0.001 | 11.9min | ✅ |
| `cifar10_ce_e50_bs2056_lr0.01` | ce | **62.96%** | 62.68% | 50 | 2056 | 0.01 | 11.9min | ✅ |
| `cifar10_hinge_b_annealing_m1.0_b1.0-100.0_e50_bs128_lr0.005` | hinge | **54.68%** | 10.00% | 50 | 128 | 0.005 | 48.7min | ❌ |
| `cifar10_hinge_beta_annealing_m1.0_beta0.5-5.0_e60_bs128_lr0.005` | hinge | **66.70%** | 65.63% | 60 | 128 | 0.005 | 58.6min | ✅ |
| `cifar10_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e60_bs128_lr0.005` | hinge | **55.30%** | 10.00% | 60 | 128 | 0.005 | 57.8min | ❌ |
| `cifar10_hinge_m1.0_e160_bs128_lr0.005` | hinge | **67.96%** | 66.68% | 160 | 128 | 0.005 | 156.2min | ✅ |
| `cifar10_vlog_annealing_b5.0_beta0.5-5.0_e160_bs128_lr0.005` | vlog | **50.91%** | 50.40% | 160 | 128 | 0.005 | 157.1min | ✅ |
| `cifar10_vlog_b_annealing_b1.0-100.0_beta1.0_e60_bs128_lr0.005` | vlog | **53.96%** | 45.45% | 60 | 128 | 0.005 | 59.1min | ❌ |
| `cifar10_vlog_fixed_b1.0_beta1.0_e50_bs128_lr0.005` | vlog | **58.76%** | 58.51% | 50 | 128 | 0.005 | 48.9min | ✅ |
| `cifar10_vlog_fixed_b2.0_beta1.0_e50_bs128_lr0.005` | vlog | **62.29%** | 62.29% | 50 | 128 | 0.005 | 48.9min | ✅ |
| `cifar10_vlog_fixed_b20.0_beta1.0_e50_bs128_lr0.005` | vlog | **37.58%** | 37.58% | 50 | 128 | 0.005 | 48.8min | ❌ |
| `cifar10_vlog_fixed_b5.0_beta1.0_e50_bs128_lr0.005` | vlog | **45.73%** | 45.21% | 50 | 128 | 0.005 | 48.8min | ❌ |

---

# MNIST

**15 experiments** (13 successful, 2 failed)

| Filename | Loss | Best Acc | Final Acc | Epochs | BS | LR | Time | Status |
|----------|------|----------|-----------|--------|----|----|------|--------|
| `mnist_ce_e10_bs512_lr0.01` | ce | **94.69%** | 94.04% | 10 | 512 | 0.01 | 2.4min | ✅ |
| `mnist_hinge_b_annealing_m1.0_b1.0-100.0_e10_bs512_lr0.01` | hinge | **92.42%** | 9.80% | 10 | 512 | 0.01 | 2.3min | ❌ |
| `mnist_hinge_beta_annealing_m1.0_beta0.5-5.0_e10_bs512_lr0.01` | hinge | **94.73%** | 94.44% | 10 | 512 | 0.01 | 2.3min | ✅ |
| `mnist_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e10_bs512_lr0.01` | hinge | **92.18%** | 9.80% | 10 | 512 | 0.01 | 2.3min | ❌ |
| `mnist_hinge_m1.0_e10_bs512_lr0.01` | hinge | **94.76%** | 94.57% | 10 | 512 | 0.01 | 2.3min | ✅ |
| `mnist_vlog_annealing_b5.0_beta0.5-100.0_e10_bs512_lr0.01` | vlog | **94.44%** | 94.17% | 10 | 512 | 0.01 | 2.4min | ✅ |
| `mnist_vlog_annealing_b5.0_beta0.5-100.0_e20_bs512_lr0.01` | vlog | **94.96%** | 94.65% | 20 | 512 | 0.01 | 4.8min | ✅ |
| `mnist_vlog_annealing_b5.0_beta0.5-5.0_e10_bs512_lr0.01` | vlog | **94.69%** | 94.38% | 10 | 512 | 0.01 | 2.4min | ✅ |
| `mnist_vlog_b_annealing_b1.0-10.0_beta1.0_e10_bs512_lr0.01` | vlog | **94.82%** | 94.64% | 10 | 512 | 0.01 | 2.3min | ✅ |
| `mnist_vlog_b_annealing_b1.0-100.0_beta1.0_e10_bs512_lr0.01` | vlog | **94.49%** | 94.18% | 10 | 512 | 0.01 | 2.3min | ✅ |
| `mnist_vlog_b_annealing_b10.0-1.0_beta1.0_e10_bs512_lr0.01` | vlog | **92.69%** | 92.69% | 10 | 512 | 0.01 | 2.3min | ✅ |
| `mnist_vlog_fixed_b1.0_beta1.0_e10_bs512_lr0.01` | vlog | **95.15%** | 95.15% | 10 | 512 | 0.01 | 2.4min | ✅ |
| `mnist_vlog_fixed_b2.0_beta1.0_e10_bs512_lr0.01` | vlog | **94.51%** | 94.51% | 10 | 512 | 0.01 | 2.4min | ✅ |
| `mnist_vlog_fixed_b20.0_beta1.0_e10_bs512_lr0.01` | vlog | **93.68%** | 93.68% | 10 | 512 | 0.01 | 2.4min | ✅ |
| `mnist_vlog_fixed_b5.0_beta1.0_e10_bs512_lr0.01` | vlog | **93.66%** | 93.18% | 10 | 512 | 0.01 | 2.4min | ✅ |

---

# Training Curves

## CIFAR-10

### `cifar10_ce_e10_bs512_lr0.0005` | 54.33% ✅

![cifar10_ce_e10_bs512_lr0.0005](../plots/cifar10_ce_e10_bs512_lr0.0005.png)

### `cifar10_ce_e10_bs512_lr0.005` | 57.29% ✅

![cifar10_ce_e10_bs512_lr0.005](../plots/cifar10_ce_e10_bs512_lr0.005.png)

### `cifar10_ce_e10_bs512_lr0.01` | 57.68% ✅

![cifar10_ce_e10_bs512_lr0.01](../plots/cifar10_ce_e10_bs512_lr0.01.png)

### `cifar10_ce_e10_bs512_lr0.1` | 45.49% ❌

![cifar10_ce_e10_bs512_lr0.1](../plots/cifar10_ce_e10_bs512_lr0.1.png)

### `cifar10_ce_e160_bs128_lr0.005` | 67.73% ✅

![cifar10_ce_e160_bs128_lr0.005](../plots/cifar10_ce_e160_bs128_lr0.005.png)

### `cifar10_ce_e20_bs2056_lr0.01` | 58.79% ✅

![cifar10_ce_e20_bs2056_lr0.01](../plots/cifar10_ce_e20_bs2056_lr0.01.png)

### `cifar10_ce_e20_bs512_lr0.01` | 61.28% ✅

![cifar10_ce_e20_bs512_lr0.01](../plots/cifar10_ce_e20_bs512_lr0.01.png)

### `cifar10_ce_e50_bs2056_lr0.0001` | 56.99% ✅

![cifar10_ce_e50_bs2056_lr0.0001](../plots/cifar10_ce_e50_bs2056_lr0.0001.png)

### `cifar10_ce_e50_bs2056_lr0.001` | 61.10% ✅

![cifar10_ce_e50_bs2056_lr0.001](../plots/cifar10_ce_e50_bs2056_lr0.001.png)

### `cifar10_ce_e50_bs2056_lr0.01` | 62.96% ✅

![cifar10_ce_e50_bs2056_lr0.01](../plots/cifar10_ce_e50_bs2056_lr0.01.png)

### `cifar10_hinge_b_annealing_m1.0_b1.0-100.0_e50_bs128_lr0.005` | 54.68% ❌

![cifar10_hinge_b_annealing_m1.0_b1.0-100.0_e50_bs128_lr0.005](../plots/cifar10_hinge_b_annealing_m1.0_b1.0-100.0_e50_bs128_lr0.005.png)

### `cifar10_hinge_beta_annealing_m1.0_beta0.5-5.0_e60_bs128_lr0.005` | 66.70% ✅

![cifar10_hinge_beta_annealing_m1.0_beta0.5-5.0_e60_bs128_lr0.005](../plots/cifar10_hinge_beta_annealing_m1.0_beta0.5-5.0_e60_bs128_lr0.005.png)

### `cifar10_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e60_bs128_lr0.005` | 55.30% ❌

![cifar10_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e60_bs128_lr0.005](../plots/cifar10_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e60_bs128_lr0.005.png)

### `cifar10_hinge_m1.0_e160_bs128_lr0.005` | 67.96% ✅

![cifar10_hinge_m1.0_e160_bs128_lr0.005](../plots/cifar10_hinge_m1.0_e160_bs128_lr0.005.png)

### `cifar10_vlog_annealing_b5.0_beta0.5-5.0_e160_bs128_lr0.005` | 50.91% ✅

![cifar10_vlog_annealing_b5.0_beta0.5-5.0_e160_bs128_lr0.005](../plots/cifar10_vlog_annealing_b5.0_beta0.5-5.0_e160_bs128_lr0.005.png)

### `cifar10_vlog_b_annealing_b1.0-100.0_beta1.0_e60_bs128_lr0.005` | 53.96% ❌

![cifar10_vlog_b_annealing_b1.0-100.0_beta1.0_e60_bs128_lr0.005](../plots/cifar10_vlog_b_annealing_b1.0-100.0_beta1.0_e60_bs128_lr0.005.png)

### `cifar10_vlog_fixed_b1.0_beta1.0_e50_bs128_lr0.005` | 58.76% ✅

![cifar10_vlog_fixed_b1.0_beta1.0_e50_bs128_lr0.005](../plots/cifar10_vlog_fixed_b1.0_beta1.0_e50_bs128_lr0.005.png)

### `cifar10_vlog_fixed_b2.0_beta1.0_e50_bs128_lr0.005` | 62.29% ✅

![cifar10_vlog_fixed_b2.0_beta1.0_e50_bs128_lr0.005](../plots/cifar10_vlog_fixed_b2.0_beta1.0_e50_bs128_lr0.005.png)

### `cifar10_vlog_fixed_b20.0_beta1.0_e50_bs128_lr0.005` | 37.58% ❌

![cifar10_vlog_fixed_b20.0_beta1.0_e50_bs128_lr0.005](../plots/cifar10_vlog_fixed_b20.0_beta1.0_e50_bs128_lr0.005.png)

### `cifar10_vlog_fixed_b5.0_beta1.0_e50_bs128_lr0.005` | 45.73% ❌

![cifar10_vlog_fixed_b5.0_beta1.0_e50_bs128_lr0.005](../plots/cifar10_vlog_fixed_b5.0_beta1.0_e50_bs128_lr0.005.png)


## MNIST

### `mnist_ce_e10_bs512_lr0.01` | 94.69% ✅

![mnist_ce_e10_bs512_lr0.01](../plots/mnist_ce_e10_bs512_lr0.01.png)

### `mnist_hinge_b_annealing_m1.0_b1.0-100.0_e10_bs512_lr0.01` | 92.42% ❌

![mnist_hinge_b_annealing_m1.0_b1.0-100.0_e10_bs512_lr0.01](../plots/mnist_hinge_b_annealing_m1.0_b1.0-100.0_e10_bs512_lr0.01.png)

### `mnist_hinge_beta_annealing_m1.0_beta0.5-5.0_e10_bs512_lr0.01` | 94.73% ✅

![mnist_hinge_beta_annealing_m1.0_beta0.5-5.0_e10_bs512_lr0.01](../plots/mnist_hinge_beta_annealing_m1.0_beta0.5-5.0_e10_bs512_lr0.01.png)

### `mnist_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e10_bs512_lr0.01` | 92.18% ❌

![mnist_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e10_bs512_lr0.01](../plots/mnist_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e10_bs512_lr0.01.png)

### `mnist_hinge_m1.0_e10_bs512_lr0.01` | 94.76% ✅

![mnist_hinge_m1.0_e10_bs512_lr0.01](../plots/mnist_hinge_m1.0_e10_bs512_lr0.01.png)

### `mnist_vlog_annealing_b5.0_beta0.5-100.0_e10_bs512_lr0.01` | 94.44% ✅

![mnist_vlog_annealing_b5.0_beta0.5-100.0_e10_bs512_lr0.01](../plots/mnist_vlog_annealing_b5.0_beta0.5-100.0_e10_bs512_lr0.01.png)

### `mnist_vlog_annealing_b5.0_beta0.5-100.0_e20_bs512_lr0.01` | 94.96% ✅

![mnist_vlog_annealing_b5.0_beta0.5-100.0_e20_bs512_lr0.01](../plots/mnist_vlog_annealing_b5.0_beta0.5-100.0_e20_bs512_lr0.01.png)

### `mnist_vlog_annealing_b5.0_beta0.5-5.0_e10_bs512_lr0.01` | 94.69% ✅

![mnist_vlog_annealing_b5.0_beta0.5-5.0_e10_bs512_lr0.01](../plots/mnist_vlog_annealing_b5.0_beta0.5-5.0_e10_bs512_lr0.01.png)

### `mnist_vlog_b_annealing_b1.0-10.0_beta1.0_e10_bs512_lr0.01` | 94.82% ✅

![mnist_vlog_b_annealing_b1.0-10.0_beta1.0_e10_bs512_lr0.01](../plots/mnist_vlog_b_annealing_b1.0-10.0_beta1.0_e10_bs512_lr0.01.png)

### `mnist_vlog_b_annealing_b1.0-100.0_beta1.0_e10_bs512_lr0.01` | 94.49% ✅

![mnist_vlog_b_annealing_b1.0-100.0_beta1.0_e10_bs512_lr0.01](../plots/mnist_vlog_b_annealing_b1.0-100.0_beta1.0_e10_bs512_lr0.01.png)

### `mnist_vlog_b_annealing_b10.0-1.0_beta1.0_e10_bs512_lr0.01` | 92.69% ✅

![mnist_vlog_b_annealing_b10.0-1.0_beta1.0_e10_bs512_lr0.01](../plots/mnist_vlog_b_annealing_b10.0-1.0_beta1.0_e10_bs512_lr0.01.png)

### `mnist_vlog_fixed_b1.0_beta1.0_e10_bs512_lr0.01` | 95.15% ✅

![mnist_vlog_fixed_b1.0_beta1.0_e10_bs512_lr0.01](../plots/mnist_vlog_fixed_b1.0_beta1.0_e10_bs512_lr0.01.png)

### `mnist_vlog_fixed_b2.0_beta1.0_e10_bs512_lr0.01` | 94.51% ✅

![mnist_vlog_fixed_b2.0_beta1.0_e10_bs512_lr0.01](../plots/mnist_vlog_fixed_b2.0_beta1.0_e10_bs512_lr0.01.png)

### `mnist_vlog_fixed_b20.0_beta1.0_e10_bs512_lr0.01` | 93.68% ✅

![mnist_vlog_fixed_b20.0_beta1.0_e10_bs512_lr0.01](../plots/mnist_vlog_fixed_b20.0_beta1.0_e10_bs512_lr0.01.png)

### `mnist_vlog_fixed_b5.0_beta1.0_e10_bs512_lr0.01` | 93.66% ✅

![mnist_vlog_fixed_b5.0_beta1.0_e10_bs512_lr0.01](../plots/mnist_vlog_fixed_b5.0_beta1.0_e10_bs512_lr0.01.png)


---

*Generated by `analyze_all_results.py`*
