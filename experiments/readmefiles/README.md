# MNIST Binary Network with MCMC-Inspired Loss

This experiment compares different loss functions for training binary neural networks on MNIST:

1. **Cross-Entropy** (baseline)
2. **Vlog potential with fixed β** 
3. **Vlog potential with β-annealing**

## Quick Start

### Run Single Experiment

```bash
# Baseline: Cross-Entropy
python experiments/mnist_mcmc_experiment.py --loss-type ce --epochs 100

# Vlog with fixed beta=1.0
python experiments/mnist_mcmc_experiment.py --loss-type vlog_fixed --beta-fixed 1.0 --b-value 10.0 --epochs 100

# Vlog with beta-annealing (0.1 -> 100.0)
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --beta-start 0.1 --beta-end 100.0 --b-value 10.0 --epochs 100
```

### Run All Experiments

```bash
python experiments/run_all_experiments.py
```

## Loss Functions

### Cross-Entropy (Baseline)
Standard multiclass cross-entropy loss.

### Vlog Potential

From MCMC perceptron solver:

```
V_log(x, b) = b * (1 - x^(1/b))  if x > 0
            = b * (1 - x)        if x <= 0
```

**Stability** (margin) for multi-class:
```
stability = (correct_class_score - max_wrong_class_score) / sqrt(N)
```

**Total Loss**:
```
L = β * mean(V_log(stabilities, b))
```

### Beta-Annealing

β increases linearly from β_start to β_end over training epochs.
- **Small β** (early training): Loss landscape is smooth, allows exploration
- **Large β** (late training): Loss focuses sharply on minimum, refines solution

This is inspired by simulated annealing but applied to gradient descent.

## Key Parameters

- `--loss-type`: Choose loss function (ce, vlog_fixed, vlog_annealing)
- `--b-value`: τ parameter in Vlog potential (default: 10.0)
- `--beta-start`: Starting β for annealing (default: 0.1)
- `--beta-end`: Ending β for annealing (default: 100.0)
- `--beta-fixed`: Fixed β when not annealing (default: 1.0)
- `--normalization-dim`: Dimension for stability normalization (default: 10)

## Results

Results are saved in `experiments/results/` with format:
- `mnist_ce_results.txt`
- `mnist_vlog_fixed_results.txt`
- `mnist_vlog_annealing_results.txt`

## Architecture

- Binary MLP: 784 → 6144 → 6144 → 6144 → 10
- Binary activations and weights (except first layer input and last layer)
- Batch normalization between layers
- Dropout (0.5) before final layer

## Notes

Based on research comparing:
- Standard gradient descent (cross-entropy)
- MCMC-inspired potentials without annealing
- MCMC-inspired potentials with β-annealing (temperature/focus control)

Goal: Determine if β-annealing improves convergence and final accuracy on binary networks.

