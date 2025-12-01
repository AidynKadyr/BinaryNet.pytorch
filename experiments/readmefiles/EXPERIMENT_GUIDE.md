# 🔬 Experiment Guide: What Are We Actually Testing?

## Current Implementation Status

### ⚠️ IMPORTANT: Beta Annealing is NOT Active!

In the current code (line 101-102 of `mnist_mcmc_experiment.py`):
```python
#loss = self.beta * potentials.mean()  ← COMMENTED OUT (causes instability)
loss = potentials.mean()                ← ACTIVE (β not used)
```

**This means β-annealing is scheduled but has NO effect on training!**

---

## 📊 What Each Experiment Actually Does

| Experiment | Loss Function | Beta (β) | b (τ) | What It Tests |
|-----------|---------------|----------|-------|---------------|
| `--loss-type ce` | Cross-Entropy | N/A | N/A | **Baseline**: Standard BNN training |
| `--loss-type vlog_fixed` | Vlog potential | Ignored | 10.0 (fixed) | **Does Vlog help?** |
| `--loss-type vlog_annealing` | Vlog potential | Ignored | 10.0 (fixed) | **SAME AS vlog_fixed!** ⚠️ |

### 🎯 Current Comparison

**What you're actually comparing:**
- ✅ Cross-Entropy vs. Vlog potential
- ❌ NOT comparing annealing effects (vlog_fixed = vlog_annealing currently)

---

## 🚀 Running Experiments

### For Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set plot directory to Google Drive
plot_dir = '/content/drive/MyDrive/BNN_Experiments/plots'

# Run experiments with plots
!python experiments/mnist_mcmc_experiment.py \
    --loss-type ce \
    --epochs 100 \
    --batch-size 512 \
    --plot-dir {plot_dir}

!python experiments/mnist_mcmc_experiment.py \
    --loss-type vlog_fixed \
    --epochs 100 \
    --batch-size 512 \
    --plot-dir {plot_dir}
```

### Quick Test (10 epochs)

```bash
# Test all three (takes ~5 minutes on GPU)
python experiments/mnist_mcmc_experiment.py --loss-type ce --epochs 10 --batch-size 512
python experiments/mnist_mcmc_experiment.py --loss-type vlog_fixed --epochs 10 --batch-size 512
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --epochs 10 --batch-size 512
```

---

## 📈 Understanding the Plots

Each experiment generates a plot with:

### Left Panel: Loss vs Epochs
- **Blue line**: Training loss
- **Red line**: Test loss
- Look for: Smooth descent, no divergence

### Right Panel: Accuracy vs Epochs
- **Blue line**: Training accuracy
- **Red line**: Test accuracy
- Look for: Final accuracy, convergence speed

### Plot Filenames
- `mnist_ce.png` - Cross-Entropy baseline
- `mnist_vlog_fixed_b10.0_beta1.0.png` - Vlog (β=1.0, b=10)
- `mnist_vlog_annealing_b10.0_beta0.1-100.0.png` - Vlog "annealing" (but β not used!)

---

## 🔧 To Make Beta Annealing Actually Work

### Option A: Anneal b (τ) parameter [RECOMMENDED]

Modify `VlogLoss.forward()` to use β for controlling b:

```python
def forward(self, output, target):
    stabilities = self.compute_stabilities(output, target)
    # Use beta to anneal b from 1 to 100
    current_b = 1.0 + (self.b - 1.0) * (self.beta / 100.0)
    potentials = self.vlog_potential_with_b(stabilities, current_b)
    loss = potentials.mean()
    return loss
```

Then:
- `vlog_fixed`: b = 10 (constant)
- `vlog_annealing`: b = 1 → 100 (annealed via β)

### Option B: Just compare CE vs Vlog

Since annealing isn't working, simplify to 2 experiments:
1. Cross-Entropy (baseline)
2. Vlog potential (b=10)

---

## 📝 Vlog Loss Explained

### What is Vlog?

From your Julia MCMC code:
```python
V_log(x, b) = b * (1 - x^(1/b))  if x > 0  # Rewards large margins
            = b * (1 - x)        if x <= 0  # Penalizes violations
```

### Multi-class Stability

For MNIST (10 classes):
```python
stability = (score_correct - score_max_wrong) / sqrt(10)
```

### Final Loss

```python
loss = mean(V_log(stability_per_sample, b=10))
```

**Key difference from Cross-Entropy:**
- CE: Optimizes log-likelihood
- Vlog: Optimizes margin-based potential (more like SVM)

---

## 🎯 What to Look For in Results

### Hypothesis: Vlog improves over Cross-Entropy

**Evidence for:**
- ✅ Higher final test accuracy
- ✅ Faster convergence
- ✅ More stable training (smoother loss curves)

**Evidence against:**
- ❌ Lower final accuracy
- ❌ Slower convergence  
- ❌ Unstable training

### Typical MNIST BNN Performance

- **Cross-Entropy**: 96-98% accuracy
- **Your Vlog (10 epochs)**: ~96% accuracy
- **Expected with 100 epochs**: 97-98%

---

## 🐛 Troubleshooting

### "Loss values are huge/negative"
- This is normal for Vlog! Large positive margins → negative loss values
- What matters: Does accuracy improve?

### "Plots not showing up"
- Check the `--plot-dir` path exists
- Look for "📊 Plot saved to: ..." message

### "Same results for vlog_fixed and vlog_annealing"
- Expected! β isn't currently used (see top of this doc)
- Both experiments are identical right now

---

## 📊 Expected Output Example

```
Training Complete!
Final Test Accuracy: 96.26%
Best Test Accuracy: 96.26% (Epoch 10)

📊 Plot saved to: experiments/plots/mnist_vlog_fixed_b10.0_beta1.0.png

Results saved to experiments/results/mnist_vlog_fixed_results.txt
```

---

## 🎓 Summary

**Current state:**
1. ✅ Plotting works - visual tracking of training
2. ✅ Vlog potential implemented correctly
3. ❌ β-annealing not functional (β scheduled but not used)
4. ❌ vlog_fixed and vlog_annealing give identical results

**Recommendation:**
Focus on comparing CE vs Vlog first. If Vlog helps, then implement proper τ-annealing.

