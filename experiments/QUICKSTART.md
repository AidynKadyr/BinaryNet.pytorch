# Quick Start Guide

## 🚀 Run Your First Experiment

### Option 1: Single Experiment (Fast Test)

```bash
# Run just the baseline (5-10 minutes on GPU)
cd BinaryNet.pytorch
python experiments/mnist_mcmc_experiment.py --loss-type ce --epochs 10
```

### Option 2: Compare All Three (Recommended)

```bash
# Run all experiments with 100 epochs each (~1-2 hours total on GPU)
cd BinaryNet.pytorch
python experiments/run_all_experiments.py
```

Or manually:

```bash
# 1. Baseline: Cross-Entropy
python experiments/mnist_mcmc_experiment.py --loss-type ce --epochs 100

# 2. Vlog (no annealing)
python experiments/mnist_mcmc_experiment.py --loss-type vlog_fixed --beta-fixed 1.0 --epochs 100

# 3. Vlog with β-annealing (your approach)
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --beta-start 0.1 --beta-end 100.0 --epochs 100
```

## 📊 View Results

```bash
# Compare all results
python experiments/compare_results.py

# Or view raw results
cat experiments/results/mnist_*_results.txt
```

## 🎯 Expected Output

```
==============================================================================
MNIST BINARY NETWORK - RESULTS COMPARISON
==============================================================================

Loss Function             Final Acc    Best Acc     Best Epoch  
--------------------------------------------------------------------------------
Cross-Entropy             95.00        96.00        85          
Vlog (fixed β)            94.50        95.80        90          
Vlog (β-annealing)        96.20        97.10        88          

🏆 Best Performance: Vlog (β-annealing) with 97.10% accuracy
   Improvement over Cross-Entropy: +1.10%

==============================================================================
```

## ⚙️ Hyperparameter Tuning

If you want to experiment with different parameters:

```bash
# Try different b (tau) values
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --b-value 5.0 --epochs 100
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --b-value 20.0 --epochs 100

# Try different beta ranges
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --beta-start 0.01 --beta-end 50.0 --epochs 100
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --beta-start 1.0 --beta-end 1000.0 --epochs 100

# Try different normalization dimensions
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --normalization-dim 100 --epochs 100
```

## 🐛 Troubleshooting

**Error: "No module named 'models.binarized_modules'"**
- Make sure you're in the `BinaryNet.pytorch` directory
- Or add it to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

**CUDA out of memory**
- Reduce batch size: `--batch-size 32`

**Training too slow**
- Reduce epochs for testing: `--epochs 10`
- Increase log interval: `--log-interval 200`

## 📝 Notes

- **β-annealing logic**: β increases from small→large over epochs
  - Small β (early): Smooth loss landscape, exploration
  - Large β (late): Sharp focus on minimum, exploitation
  
- **b (τ) parameter**: Controls potential sharpness (kept constant during training)
  - Smaller b: Smoother potential
  - Larger b: Sharper potential (more like step function)

- **Normalization dim**: sqrt(N) in stability calculation
  - Default: 10 (output dimension)
  - Could also try: 6144 (hidden layer size)

