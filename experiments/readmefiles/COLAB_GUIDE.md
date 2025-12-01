# 🎓 Google Colab Quick Start Guide

## 🚀 Running Experiments in Colab

### Step 1: Clone Repository

```python
!git clone https://github.com/YOUR_USERNAME/BinaryNet.pytorch.git
%cd BinaryNet.pytorch
```

### Step 2: Install Dependencies (if needed)

```python
# Usually not needed, but just in case:
!pip install torch torchvision matplotlib numpy
```

### Step 3: Mount Google Drive (Optional but Recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Run Experiments

#### Option A: Quick Interactive Runner

```python
!python experiments/colab_runner.py
```

This will:
- Automatically detect Colab
- Mount Google Drive
- Ask which experiments to run
- Save plots and results to Drive

#### Option B: Manual Commands

```python
# Set plot directory (use Google Drive to save plots)
plot_dir = '/content/drive/MyDrive/BNN_Experiments/plots'

# Experiment 1: Cross-Entropy baseline
!python experiments/mnist_mcmc_experiment.py \
    --loss-type ce \
    --epochs 10 \
    --batch-size 512 \
    --plot-dir {plot_dir}

# Experiment 2: Vlog potential
!python experiments/mnist_mcmc_experiment.py \
    --loss-type vlog_fixed \
    --epochs 10 \
    --batch-size 512 \
    --plot-dir {plot_dir}
```

### Step 5: View Results

```python
# View plots inline
from IPython.display import Image, display
import glob

# List all generated plots
plots = glob.glob(f'{plot_dir}/*.png')
for plot in sorted(plots):
    print(f"\n📊 {plot}")
    display(Image(filename=plot, width=800))
```

---

## 📊 Understanding Your Results

### Plot Files

After running experiments, you'll get plots like:
- `mnist_ce.png` - Cross-Entropy baseline
- `mnist_vlog_fixed_b10.0_beta1.0.png` - Vlog potential

Each plot shows:
- **Left panel**: Loss curves (training vs test)
- **Right panel**: Accuracy curves (training vs test)
- **Title**: Experiment configuration

### What to Look For

✅ **Good signs:**
- Test accuracy > 95%
- Smooth loss curves (no wild oscillations)
- Test accuracy improving over epochs

❌ **Bad signs:**
- Test accuracy < 90%
- Loss exploding/diverging
- Large gap between train and test accuracy (overfitting)

---

## 🎯 Typical Results (10 epochs)

| Loss Type | Final Test Acc | Training Time (GPU) |
|-----------|----------------|---------------------|
| Cross-Entropy | 96-97% | ~2 min |
| Vlog | 95-96% | ~2 min |

**With 100 epochs:** Both should reach 97-98%

---

## 🔧 Common Issues

### 1. "RuntimeError: CUDA out of memory"

```python
# Reduce batch size
!python experiments/mnist_mcmc_experiment.py \
    --loss-type ce \
    --epochs 10 \
    --batch-size 256  # Reduced from 512
```

### 2. "ModuleNotFoundError: No module named 'models'"

```python
# Make sure you're in the right directory
%cd BinaryNet.pytorch
!pwd  # Should show .../BinaryNet.pytorch
```

### 3. Plots not saving to Drive

```python
# Make sure Drive is mounted
import os
print(os.path.exists('/content/drive/MyDrive'))  # Should print True

# Create directory manually
!mkdir -p /content/drive/MyDrive/BNN_Experiments/plots
```

---

## 📈 Quick Comparison Script

```python
# After running experiments, compare results
import pandas as pd
import matplotlib.pyplot as plt

# Load results
ce_acc = [...]  # Get from mnist_ce_results.txt
vlog_acc = [...]  # Get from mnist_vlog_fixed_results.txt

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(ce_acc, label='Cross-Entropy', linewidth=2)
plt.plot(vlog_acc, label='Vlog', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Cross-Entropy vs Vlog Potential')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 💡 Pro Tips

1. **Start with 10 epochs** for quick tests, then run 100 epochs for final results
2. **Use Google Drive** to persist results across Colab sessions
3. **Save your notebook** to Drive before closing Colab
4. **Check GPU allocation**: Runtime → Change runtime type → GPU

---

## 🎓 Full Workflow Example

```python
# 1. Setup
!git clone https://github.com/YOUR_USERNAME/BinaryNet.pytorch.git
%cd BinaryNet.pytorch

from google.colab import drive
drive.mount('/content/drive')

plot_dir = '/content/drive/MyDrive/BNN_Experiments/plots'

# 2. Quick test (10 epochs)
!python experiments/mnist_mcmc_experiment.py --loss-type ce --epochs 10 --batch-size 512 --plot-dir {plot_dir}
!python experiments/mnist_mcmc_experiment.py --loss-type vlog_fixed --epochs 10 --batch-size 512 --plot-dir {plot_dir}

# 3. View plots
from IPython.display import Image, display
import glob

plots = glob.glob(f'{plot_dir}/*.png')
for plot in sorted(plots):
    print(f"\n{plot}")
    display(Image(filename=plot, width=800))

# 4. If results look good, run full training
!python experiments/mnist_mcmc_experiment.py --loss-type ce --epochs 100 --batch-size 512 --plot-dir {plot_dir}
!python experiments/mnist_mcmc_experiment.py --loss-type vlog_fixed --epochs 100 --batch-size 512 --plot-dir {plot_dir}
```

---

## 📚 Next Steps

1. Run baseline comparison (CE vs Vlog)
2. Check EXPERIMENT_GUIDE.md for understanding what's being tested
3. If Vlog works well, consider implementing proper τ-annealing
4. Try different b values: `--b-value 5.0` or `--b-value 20.0`

---

**Questions?** See EXPERIMENT_GUIDE.md for detailed explanations!

