# 🚀 HingeLoss with b-Annealing and β-Annealing

## Overview

This implementation fulfills **Point 5** from your notes:
> "Compare tau-annealing vs simulated annealing in temperature on the hinge loss"

We've implemented **THREE** annealing strategies for HingeLoss:
1. **b-annealing** (τ-annealing): Sharpen the loss landscape
2. **β-annealing**: Scale the loss magnitude  
3. **Both simultaneously**: Combined approach

---

## 📊 Complete Loss Function Catalog

| Loss Type | Description | Annealing | Command Flag |
|-----------|-------------|-----------|--------------|
| **Cross-Entropy** | Standard softmax-based | ❌ None | `--loss-type ce` |
| **Hinge** | Standard SVM-style | ❌ None | `--loss-type hinge` |
| **Hinge + b-Annealing** | Sharpness annealing | ✅ b: 1→100 | `--loss-type hinge_b_annealing` |
| **Hinge + β-Annealing** | Temperature annealing | ✅ β: 0.5→5 | `--loss-type hinge_beta_annealing` |
| **Hinge + BOTH** | Combined annealing | ✅ b + β | `--loss-type hinge_both_annealing` |
| **Vlog Fixed** | MCMC potential | ❌ None | `--loss-type vlog_fixed` |
| **Vlog + β-Annealing** | MCMC + temperature | ✅ β: 0.5→5 | `--loss-type vlog_annealing` |
| **Vlog + b-Annealing** | MCMC + sharpness | ✅ b: 1→100 | `--loss-type vlog_b_annealing` |
| **Vlog + BOTH** | MCMC + combined | ✅ b + β | `--loss-type vlog_both_annealing` |

---

## 🔬 What Each Annealing Strategy Does

### 1️⃣ **b-Annealing (τ-annealing)**
```python
Loss = β × mean(max(0, margin - y*f(x))^b)
                                      ^^^
                                   b increases
```
- **b starts at 1.0**: Standard hinge loss
- **b increases to 100**: Loss becomes sharper (more step-function-like)
- **Schedule**: Exponential growth (like Julia MCMC)
- **Interpretation**: Gradually make the loss less forgiving of mistakes

### 2️⃣ **β-Annealing (Temperature Annealing)**
```python
Loss = β × mean(max(0, margin - y*f(x)))
       ^
    β increases
```
- **β starts at 0.5**: Low loss magnitude (exploration)
- **β increases to 5.0**: High loss magnitude (exploitation)
- **Schedule**: Linear growth
- **Interpretation**: Simulated annealing in gradient descent

### 3️⃣ **Both Annealing**
```python
Loss = β × mean(max(0, margin - y*f(x))^b)
       ^                              ^
    increases                    increases
```
- Combines both strategies
- Tests if they complement each other

---

## 🎯 Usage Examples

### Quick 10-Epoch Tests (Google Colab)

```python
plot_dir = '/content/drive/MyDrive/2. Bocconi/THESIS/BinaryNet.pytorch/experiments/plots'

# Baseline: Standard Hinge (no annealing)
!python experiments/mnist_mcmc_experiment.py \
    --loss-type hinge \
    --epochs 10 \
    --batch-size 4096 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"

# Test 1: b-Annealing (τ-annealing)
!python experiments/mnist_mcmc_experiment.py \
    --loss-type hinge_b_annealing \
    --hinge-b-start 1.0 \
    --hinge-b-end 100.0 \
    --epochs 10 \
    --batch-size 4096 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"

# Test 2: β-Annealing (temperature)
!python experiments/mnist_mcmc_experiment.py \
    --loss-type hinge_beta_annealing \
    --hinge-beta-start 0.5 \
    --hinge-beta-end 5.0 \
    --epochs 10 \
    --batch-size 4096 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"

# Test 3: BOTH (combined)
!python experiments/mnist_mcmc_experiment.py \
    --loss-type hinge_both_annealing \
    --hinge-b-start 1.0 \
    --hinge-b-end 100.0 \
    --hinge-beta-start 0.5 \
    --hinge-beta-end 5.0 \
    --epochs 10 \
    --batch-size 4096 \
    --num-workers 4 \
    --plot-dir "{plot_dir}"
```

### Custom Annealing Ranges

```bash
# Aggressive b-annealing (like Julia MCMC: 1 → 10^6)
!python experiments/mnist_mcmc_experiment.py \
    --loss-type hinge_b_annealing \
    --hinge-b-start 1.0 \
    --hinge-b-end 1000000.0 \
    --epochs 100

# Gentle β-annealing
!python experiments/mnist_mcmc_experiment.py \
    --loss-type hinge_beta_annealing \
    --hinge-beta-start 0.1 \
    --hinge-beta-end 2.0 \
    --epochs 100
```

---

## 📈 Expected Results Timeline

Based on our previous experiments:

| Loss Function | Expected 10-Epoch Accuracy | Notes |
|--------------|----------------------------|-------|
| **Cross-Entropy** | **95.64%** | Baseline |
| **Hinge (standard)** | **~95-96%** | Similar to CE |
| **Hinge + b-Annealing** | **~96-97%** 🤔 | Unknown! |
| **Hinge + β-Annealing** | **~96-97%** 🤔 | Unknown! |
| **Hinge + BOTH** | **~96-98%** 🤔 | Unknown! |
| **Vlog + β-Annealing** | **96.70%** 🏆 | Current winner |

---

## 🔬 Scientific Questions to Answer

### Key Comparison (Point 5):
**Q1: Which annealing strategy is better for HingeLoss?**
- b-annealing (sharpen landscape)
- β-annealing (scale loss)
- Both together
- Neither (standard hinge)

**Q2: Does HingeLoss + annealing match Vlog + β-annealing?**
- If yes → Annealing strategy matters more than loss function
- If no → Vlog's specific formulation is important

**Q3: Do b and β interact positively or negatively?**
- Test with `hinge_both_annealing`

---

## 🎛️ All Hyperparameters

### HingeLoss Specific
```bash
--hinge-margin 1.0              # SVM margin (default: 1.0)

# b-annealing
--hinge-b-start 1.0             # Starting b (default: 1.0)
--hinge-b-end 100.0             # Ending b (default: 100.0)

# β-annealing
--hinge-beta-start 0.5          # Starting β (default: 0.5)
--hinge-beta-end 5.0            # Ending β (default: 5.0)
```

### General Training
```bash
--epochs 100                    # Number of epochs
--batch-size 4096               # Batch size
--lr 0.01                       # Learning rate
--num-workers 4                 # Data loading workers
--plot-dir PATH                 # Where to save plots
```

---

## 📁 Output Files

All experiments generate unique filenames based on all parameters:

### Examples:
```
mnist_hinge_m1.0_e10_bs4096_lr0.01.png
mnist_hinge_b_annealing_m1.0_b1.0-100.0_e10_bs4096_lr0.01.png
mnist_hinge_beta_annealing_m1.0_beta0.5-5.0_e10_bs4096_lr0.01.png
mnist_hinge_both_annealing_m1.0_b1.0-100.0_beta0.5-5.0_e10_bs4096_lr0.01.png
```

### Results include:
- **Plots**: Loss and accuracy curves
- **Text files**: Detailed per-epoch results
- **Training time**: Total and per-epoch

---

## 🚀 Recommended Experiment Sequence

### Phase 1: Quick 10-Epoch Scan
Run all 4 HingeLoss variants with 10 epochs to see which is most promising.

### Phase 2: Deep Dive
Take the best performer(s) and run 100 epochs with different hyperparameters.

### Phase 3: Compare to Vlog
Compare best HingeLoss result to Vlog + β-annealing (96.70%).

### Phase 4: Report
Answer Point 5: Which annealing strategy (τ vs temperature) works better?

---

## 🔍 Implementation Details

### HingeLoss Class
```python
class HingeLoss(nn.Module):
    def __init__(self, margin=1.0, b=1.0, beta=1.0):
        # b controls sharpness
        # beta controls magnitude
    
    def forward(self, input, target_onehot):
        output = margin - input * target_onehot
        output = max(0, output)  # ReLU
        
        if b != 1.0:
            output = output^b  # Sharpening
        
        loss = beta * output.mean()  # Scaling
        return loss
```

### Schedulers
```python
# Exponential growth (default for b)
b = b_start * (b_end / b_start) ^ (epoch / total_epochs)

# Linear growth (default for β)
β = beta_start + (beta_end - beta_start) * (epoch / total_epochs)
```

---

## 🎓 Connection to Theory

### From Julia MCMC:
- **b-annealing**: Mimics Julia's τ-annealing (1 → 10^-6)
- **Purpose**: Make the constraint satisfaction problem harder over time

### Simulated Annealing:
- **β-annealing**: Classic simulated annealing in gradient descent
- **Purpose**: Broad exploration → focused exploitation

### Point 5 Research:
Test which approach transfers better from MCMC to gradient descent!

---

## 💡 Tips

1. **Start with 10 epochs**: Quick tests to find promising directions
2. **Watch the loss curve**: Negative loss is OK for margin-based losses
3. **Compare test accuracy**: Final metric that matters
4. **Training time**: All should be ~2.5 min for 10 epochs @ bs=4096

---

## 📚 Related Files

- `mnist_mcmc_experiment.py`: Main experiment script
- `README.md`: General framework documentation
- `EXPERIMENT_GUIDE.md`: All loss functions explained
- `COLAB_GUIDE.md`: Google Colab setup

---

**Ready to discover which annealing strategy works best! 🚀**

