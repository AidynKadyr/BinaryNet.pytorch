# BinaryNet.pytorch — Binarized Neural Networks + Physics‑Inspired (MCMC‑Style) Losses

This folder contains two tightly-related codebases:

1. **Baseline BNN training framework** (derived from the original Binarized Neural Networks work and the `convNet.pytorch` training skeleton):
   - Entry points: `main_binary.py`, `main_binary_hinge.py`, `main_mnist.py`
   - Model zoo: `models/` (binary ResNet/VGG/AlexNet variants + binarized layers)
   - Training utilities: `utils.py`, `preprocess.py`, `data.py`

2. **Thesis experiments: Physics-inspired surrogate losses and annealing schedules** (added under `experiments/`):
   - Core experimental scripts:
     - `experiments/mnist_mcmc_experiment.py`
     - `experiments/cifar10_mcmc_experiment.py`
   - Experiment runners and analysis:
     - `experiments/run_all_experiments.py`
     - `experiments/compare_results.py`
     - `experiments/summary_results/analyze_all_results.py`
   - Output artifacts:
     - `experiments/plots/` (training curves)
     - `experiments/results/` (text reports)

The core thesis question addressed in this folder is:

> Can we import ideas from **statistical physics** (in particular **annealing** and the MCMC “Vlog” potential used for binary perceptrons) into **gradient-based training** of **binary neural networks** by replacing the standard loss with an annealed, margin-based potential?

---

## 1) High-level view of what is implemented

### 1.1 Baseline: Binarized Neural Networks (BNN)
The baseline BNN implementation uses:
- **Binary activations/weights** (mostly ±1) for hidden layers,
- **HardTanh** as activation (clipping into [-1, 1]),
- **BatchNorm** heavily (stabilizes training),
- **Straight Through Estimator (STE)** for gradients through the binarization operation,
- A training loop that preserves a *real-valued shadow copy* of parameters (the `.org` fields) and uses binarized versions for the forward pass.

### 1.2 Thesis extension: MCMC-inspired potentials as losses
In `experiments/`, you implement loss functions meant to mimic the energy/potential used in the Julia MCMC perceptron work:

- **Cross Entropy (CE)** baseline (standard deep learning).
- **Hinge loss** (SVM-like margin loss) as a simpler “margin energy”.
- **Vlog loss**: a piecewise potential function designed to behave like the **log-barrier** style potential used in MCMC analyses of binary perceptrons.

The key idea is to treat training as minimizing an “energy”:
- In MCMC: sampling via `exp(-β E(w))`
- In GD: minimizing a surrogate loss `L(w) ≈ β * E(w)` with schedules that emulate annealing.

---

## 2) Directory map (what each file/folder does)

### Top-level scripts
- `main_binary.py`  
  Generic training entry point for ImageNet/CIFAR-style training using model definitions in `models/`, data from `data.py`, transforms from `preprocess.py`, and logging/checkpointing from `utils.py`.

- `main_binary_hinge.py`  
  Variant of the baseline training loop that swaps the criterion to **HingeLoss** from `models/binarized_modules.py`. It also contains explicit one-hot encoding logic for hinge training.

- `main_mnist.py`  
  A MNIST-specific baseline BNN example with its own `Net`, `train`, and `test`.

### Shared infrastructure
- `models/`  
  Binary model architectures and binarized layers:
  - `binarized_modules.py` (core binarization ops and binarized linear/conv)
  - `resnet_binary.py`, `vgg_cifar10_binary.py`, `alexnet_binary.py`
  - plus non-binary counterparts (`resnet.py`, `vgg_cifar10.py`, `alexnet.py`)

- `utils.py`  
  Logging, metrics, checkpointing, and optimizer scheduling.

- `preprocess.py`  
  Dataset transforms and augmentation.

- `data.py`  
  Dataset loaders / dataset paths (CIFAR, ImageNet, MNIST paths) used by `main_binary.py` / `main_binary_hinge.py`.  
  **Important**: the thesis scripts in `experiments/*.py` do **not** use this file; they use `torchvision.datasets.*('../data', download=True, ...)` instead.

### Thesis experiments (most important for your dissertation narrative)
- `experiments/mnist_mcmc_experiment.py`  
  A fully self-contained experiment script that:
  - defines Hinge and Vlog losses (with annealing),
  - defines a MNIST binary MLP-like network using `BinarizeLinear`,
  - trains under multiple loss types,
  - produces plots and `results/*.txt` summaries.

- `experiments/cifar10_mcmc_experiment.py`  
  CIFAR-10 equivalent pipeline:
  - uses a VGG-style binary model,
  - reuses the loss implementations from `mnist_mcmc_experiment.py`,
  - outputs plots and results similarly.

- `experiments/run_all_experiments.py`  
  Convenience script to run multiple configurations sequentially.

- `experiments/compare_results.py`  
  Reads multiple result `.txt` files and prints comparisons.

- `experiments/summary_results/analyze_all_results.py`  
  Full results aggregation and reporting:
  - collects all `.txt` results,
  - builds tables,
  - outputs Markdown + HTML reports.

- `experiments/readmefiles/`  
  Human-written guides (how to run; what was changed; command lists; etc.).

---

## 3) Core mechanism: binarization, STE, and “shadow weights”

The central trick of BNN training here is implemented in `models/binarized_modules.py`.

### 3.1 `class Binarize(InplaceFunction)`
**Purpose:** Convert a tensor to ±1 while allowing gradients to pass (STE).

- **Forward**
  - If `quant_mode == 'det'`: returns `sign(x)` (optionally scaled)
  - Else: stochastic binarization (adds noise, clamps, rounds)
  - `allow_scale=True` optionally uses `scale = max(abs(x))` to preserve magnitude

- **Backward**
  - Returns `grad_input = grad_output` (Straight Through Estimator)
  - This is the key: gradients ignore the sign non-differentiability.

### 3.2 `class Quantize(InplaceFunction)`
General quantization to `numBits` with STE in backward. (Not the thesis focus here, but included.)

### 3.3 `binarized(...)` and `quantize(...)`
Convenience wrappers calling `.apply(...)` on the autograd functions.

### 3.4 `class BinarizeLinear(nn.Linear)`
**Purpose:** A linear layer that binarizes:
- the input (except first layer with 784 features, so MNIST pixels aren’t binarized),
- the weights (always binarized from a real-valued shadow copy).

Key implementation pattern:
- On the first call, it creates `self.weight.org = self.weight.data.clone()`.
- Each forward pass uses `weight_b = binarized(self.weight.org)`.

So training conceptually maintains:
- **real-valued parameters** (optimizer updates these),
- **binary parameters** (used in forward computation).

### 3.5 `class BinarizeConv2d(nn.Conv2d)`
Same pattern as `BinarizeLinear`, but for conv kernels.
- It binarizes the input unless it is the first layer with 3 channels (RGB).
- It maintains `weight.org` shadow copy and uses binarized weights for conv.

### Why this matters for the thesis
This architecture makes binary weights trainable using standard optimizers (SGD/Adam) while still evaluating a forward pass with binary arithmetic—precisely the bridge needed between discrete physics models and gradient-based deep learning.

---

## 4) Baseline training engine (`main_binary.py`)

`main_binary.py` is a “general training runner” using the model zoo.

### 4.1 Command-line arguments
Key args:
- `--dataset {imagenet,cifar10,cifar100,...}`
- `--model` chooses from `models.__dict__` exported constructors
- `--epochs`, `--batch-size`, `--lr`, `--momentum`, `--weight-decay`
- `--resume` and `--evaluate` for checkpoint usage
- `--results_dir` and `--save` control output folder

### 4.2 `main()`
Core steps:
1. Parse args, choose output folder, set up logging (`setup_logging`).
2. Configure CUDA devices / `cudnn.benchmark`.
3. Instantiate model from `models` registry.
4. Load checkpoint optionally.
5. Create train and val transforms (`preprocess.get_transform`).
6. Build datasets (`data.get_dataset`) and `DataLoader`s.
7. Create criterion (default: `nn.CrossEntropyLoss`).
8. Create optimizer (default: SGD; can be changed by regime).
9. Epoch loop:
   - `adjust_optimizer` applies schedule changes (lr / optimizer type).
   - `train(...)` and `validate(...)`.
   - `save_checkpoint(...)`.
   - `ResultsLog.add(...)` and `.save()` (CSV).

### 4.3 `forward(...)`
Shared logic for train and validation:
- Tracks timing via `AverageMeter`.
- Runs forward pass and computes loss.
- Computes `top1/top5` accuracy via `utils.accuracy`.
- Backprop + optimizer step in training mode.

---

## 5) Thesis experimental engine (`experiments/mnist_mcmc_experiment.py`)

This file is the “workhorse” of your thesis experiments inside this repository.

It is explicitly designed to compare:

1. `ce` — Cross Entropy baseline
2. `hinge` — margin-based loss baseline
3. `hinge_b_annealing` — hinge with b (tau) annealed
4. `hinge_beta_annealing` — hinge with β annealed
5. `hinge_both_annealing` — hinge with both annealed
6. `vlog_fixed` — Vlog loss with fixed β, fixed b
7. `vlog_annealing` — Vlog loss with β annealed, b fixed
8. `vlog_b_annealing` — Vlog with b annealed, β fixed
9. `vlog_both_annealing` — Vlog with both annealed

### 5.1 `class HingeLoss(nn.Module)` (experiment-local)
This is **not** the same as `models.binarized_modules.HingeLoss`.  
This implementation adds **two annealable parameters**:

- `margin` (default 1.0)
- `b` controls sharpness: `max(0, margin - y f(x))^b`
- `beta` scales overall loss: `beta * mean(...)`

Key methods:
- `forward(input, target_onehot)`:
  - computes hinge residual `margin - input * y`
  - clamps at 0
  - optionally exponentiates by `b`
  - multiplies by `beta` and averages
- `update_b(new_b)` / `update_beta(new_beta)`:
  - used by schedulers per epoch

### 5.2 `class VlogLoss(nn.Module)`
This is the thesis-specific implementation of the MCMC potential.

#### 5.2.1 Stability for multi-class classification
For each sample, define a margin (stability) using logits:
- `correct_score = output[batch_idx, target]`
- `max_wrong_score = max(output[batch_idx, j != target])`
- `margin = (correct_score - max_wrong_score) / sqrt(normalization_dim)`

This adapts the perceptron “stability” concept to multi-class classification.

#### 5.2.2 The Vlog potential
The piecewise potential implemented is:

- If `x > 0`: `V = b * (1 - x^(1/b))`
- If `x <= 0`: `V = b * (1 - x)`

Then:
- `loss = beta * mean(V(stability))`

Methods:
- `vlog_potential(x)`: element-wise piecewise potential
- `compute_stabilities(output, target)`: correct-minus-max-wrong margin
- `forward(output, target)`: compute stability → potential → scale by β
- `update_beta`, `update_b`: used for annealing

### 5.3 `class BetaScheduler`
Produces a sequence β(t) from `beta_start` to `beta_end`:
- linear: `beta_start + (beta_end-beta_start)*t/T`
- exponential: `beta_start * (beta_end/beta_start)^(t/T)`

### 5.4 `class BScheduler`
Produces a sequence b(t) from `b_start` to `b_end`:
- linear or exponential schedules (exponential by default)

### 5.5 `class Net(nn.Module)`
MNIST network architecture:
- Flattens 28×28 → 784
- Three binarized hidden layers using `BinarizeLinear`
- BatchNorm + Hardtanh after each
- Dropout before final stage
- Final linear layer outputs 10 logits
- `forward(x, return_logits=False)`:
  - if `return_logits=True`: return raw logits (needed for CE/Vlog/Hinge)
  - else returns `LogSoftmax` output (legacy behavior)

### 5.6 `plot_training_curves(...)`
Generates and saves a 2-panel plot:
- train/test loss vs epoch
- train/test accuracy vs epoch
The function also builds an informative title encoding the experiment configuration.

### 5.7 `train(...)`
Per epoch:
1. Sets `model.train()`.
2. If schedulers exist:
   - update criterion.beta via `beta_scheduler.get_beta(epoch-1)`
   - update criterion.b via `b_scheduler.get_b(epoch-1)`
3. For each batch:
   - `output = model(data, return_logits=True)`
   - If HingeLoss: build one-hot targets with {-1, +1}
   - Compute loss; `loss.backward()`
4. **BNN weight-update logic**:
   - For each parameter `p` with `p.org`:
     - copy `p.data <- p.org` before optimizer step
   - `optimizer.step()`
   - For each parameter `p` with `p.org`:
     - clamp the updated parameter into [-1, 1] and store into `p.org`
This preserves the real-valued shadow weights while using binarized weights in forward.
5. Computes training accuracy using `argmax` on logits.

Returns:
- average training loss
- training accuracy %

### 5.8 `test(...)`
Evaluation mode:
- Same logits path
- Same hinge one-hot if hinge
- Accumulates loss and accuracy
Returns:
- average test loss (normalized by number of batches)
- test accuracy %

### 5.9 `main()`
Orchestrates the whole MNIST experiment:
- Parses CLI args for:
  - dataset loader settings
  - optimization hyperparameters
  - loss type selection
  - annealing schedule parameters for b and β
  - plot output directory
- Creates DataLoaders using torchvision MNIST.
- Instantiates model and optimizer.
- Chooses criterion and optionally schedulers based on `--loss-type`.
- Runs epochs:
  - optional LR decay every 40 epochs
  - calls `train()` and `test()`
- Saves:
  - plot in `experiments/plots/`
  - text report in `experiments/results/`
Report contains configuration, training time, and per-epoch metrics.

---

## 6) CIFAR-10 thesis experiments (`experiments/cifar10_mcmc_experiment.py`)

This mirrors the MNIST experiment but changes:
- **Data**: CIFAR-10 with standard augmentations (random crop + flip).
- **Model**: `VGG_Cifar10_Logits`, derived from `models/vgg_cifar10_binary.py` but modified to optionally return logits without applying LogSoftmax.

### 6.1 `class VGG_Cifar10_Logits(VGG_Cifar10)`
Replaces the original classifier head to return raw logits when needed.

### 6.2 `train(...)` and `test(...)`
Same structure as MNIST version:
- updates beta/b schedules
- uses hinge one-hot targets when hinge
- uses BNN shadow weight update logic (`p.org`)

### 6.3 `main()`
- builds CIFAR loaders
- runs training epochs with an LR schedule (hard-coded epoch milestones)
- saves plots and results with a naming convention encoding hyperparameters.

---

## 7) Result parsing and reporting (`experiments/summary_results/analyze_all_results.py`)

This script is your “meta-analysis” layer.

### 7.1 `class ExperimentResult`
A structured record representing one run, typically parsed from one `.txt` file in `experiments/results/`.

### 7.2 `collect_all_results(results_dir)`
Walks the results directory and parses runs into `ExperimentResult` objects.

### 7.3 `generate_summary_table(results)`
Produces table summaries (e.g., best accuracy per configuration).

### 7.4 `group_by_dataset(results)`
Separates MNIST vs CIFAR runs.

### 7.5 `get_experiment_config_key(result)`
Builds a key identifying a configuration (loss type, schedules, etc.) so that runs can be aggregated.

### 7.6 `find_cross_dataset_comparisons(results)`
Attempts to align “equivalent” experiments across datasets and compare patterns.

### 7.7 `generate_markdown_report(...)` / `generate_html_report(...)`
Writes an analysis report with tables and narrative.

### 7.8 `print_console_summary(results)`
Quick terminal summary.

### 7.9 `main()`
CLI entrypoint: run the full analysis pipeline.

---

## 8) What experiments are conducted (and why)

### 8.1 Baseline experiments (deep learning)
- CE on MNIST and CIFAR-10 using binary networks:
  - Provides standard performance reference.

### 8.2 Margin-based experiments
- Hinge loss variants:
  - Replaces probabilistic CE with a “constraint satisfaction” perspective.
  - Fits the physics analogy: a stability/margin is like the perceptron stability.

### 8.3 Physics-inspired experiments
- Vlog loss:
  - Implements a potential inspired by the MCMC energy used in your Julia work.
  - Uses the margin (correct vs best incorrect) as the stability variable.

### 8.4 Annealing experiments
Two annealing knobs:
- **β-annealing**:
  - scales the loss magnitude over training
  - interpreted as increasing “inverse temperature” → focusing the optimization
- **b-annealing** (tau-annealing analogue):
  - sharpens the potential shape over training
  - intended to mimic the `b` schedule in the Julia MCMC algorithm

And combinations:
- fixed b, anneal β
- fixed β, anneal b
- anneal both

### 8.5 Outputs
Each run produces:
- A plot (`experiments/plots/*.png`) showing training curves.
- A report (`experiments/results/*.txt`) containing configuration and per-epoch metrics.
Optionally, aggregated reports (`experiments/summary_results/RESULTS_ANALYSIS.*`).

---

## 9) Practical “how to run” (minimal)

### 9.1 MNIST
From `BinaryNet.pytorch/`:

- Cross-entropy baseline:

```bash
python experiments/mnist_mcmc_experiment.py --loss-type ce --epochs 10 --batch-size 512 --lr 0.01 --num-workers 0
```

- Vlog (fixed β, fixed b):

```bash
python experiments/mnist_mcmc_experiment.py --loss-type vlog_fixed --epochs 10 --batch-size 512 --lr 0.01 --b-value 10 --beta-fixed 1.0 --num-workers 0
```

- Vlog (β‑annealing):

```bash
python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --epochs 20 --batch-size 512 --lr 0.01 --b-value 10 --beta-start 0.5 --beta-end 5.0 --num-workers 0
```

- Hinge (baseline):

```bash
python experiments/mnist_mcmc_experiment.py --loss-type hinge --epochs 10 --batch-size 512 --lr 0.01 --hinge-margin 1.0 --num-workers 0
```

- Hinge (b‑annealing, i.e. sharpening):

```bash
python experiments/mnist_mcmc_experiment.py --loss-type hinge_b_annealing --epochs 20 --batch-size 512 --lr 0.01 --hinge-margin 1.0 --hinge-b-start 1.0 --hinge-b-end 100.0 --num-workers 0
```

- Hinge (β‑annealing, i.e. scaling):

```bash
python experiments/mnist_mcmc_experiment.py --loss-type hinge_beta_annealing --epochs 20 --batch-size 512 --lr 0.01 --hinge-margin 1.0 --hinge-beta-start 0.5 --hinge-beta-end 5.0 --num-workers 0
```

### 9.2 CIFAR‑10
From `BinaryNet.pytorch/`:

- Cross-entropy baseline:

```bash
python experiments/cifar10_mcmc_experiment.py --loss-type ce --epochs 160 --batch-size 128 --lr 0.005 --num-workers 0
```

- Vlog (β‑annealing):

```bash
python experiments/cifar10_mcmc_experiment.py --loss-type vlog_annealing --epochs 160 --batch-size 128 --lr 0.005 --b-value 5 --beta-start 0.5 --beta-end 5.0 --num-workers 0
```

- Hinge (both b‑annealing and β‑annealing):

```bash
python experiments/cifar10_mcmc_experiment.py --loss-type hinge_both_annealing --epochs 60 --batch-size 128 --lr 0.005 --hinge-margin 1.0 --hinge-b-start 1.0 --hinge-b-end 100.0 --hinge-beta-start 0.5 --hinge-beta-end 5.0 --num-workers 0
```

### 9.3 Run a preset suite of experiments
From `BinaryNet.pytorch/`:

```bash
python experiments/run_all_experiments.py
```

This script shells out to `experiments/mnist_mcmc_experiment.py` with a few preselected configurations, writing outputs to `experiments/results/` and `experiments/plots/`.

### 9.4 Aggregate and analyze results across runs
From `BinaryNet.pytorch/experiments/summary_results/`:

```bash
python analyze_all_results.py
```

Or from `BinaryNet.pytorch/experiments/`:

```bash
python summary_results/analyze_all_results.py
```

This generates:
- `experiments/summary_results/RESULTS_ANALYSIS.md`
- `experiments/summary_results/RESULTS_ANALYSIS.html`

---

## 10) Outputs and naming conventions (thesis experiments)

### 10.1 Where outputs go
- **Plots**: `experiments/plots/*.png`  
  Each plot contains loss/accuracy curves for one run.
- **Per-run logs**: `experiments/results/*.txt`  
  Each text file contains the configuration + per-epoch metrics.
- **Aggregated report**: `experiments/summary_results/RESULTS_ANALYSIS.{md,html}`

### 10.2 File naming encodes the experimental condition
Both MNIST and CIFAR scripts construct an `experiment_name` that encodes:
- dataset prefix: `mnist_...` or `cifar10_...`
- loss type: `ce`, `hinge`, `hinge_beta_annealing`, `vlog_fixed`, etc.
- annealing parameters (when relevant): `b{start}-{end}`, `beta{start}-{end}`
- training hyperparameters: `_e{epochs}_bs{batch_size}_lr{lr}`

This makes it possible to compare runs by filename alone, and lets `analyze_all_results.py` infer dataset and configuration from the text reports.

---

## 11) Dependencies (what you actually need installed)

### 11.1 For the thesis experiments (`experiments/*.py`)
These scripts rely on:
- `torch`, `torchvision`
- `numpy`
- `matplotlib`

They do **not** require `pandas` or `bokeh`.

### 11.2 For the legacy/baseline training scripts
The baseline training framework (`main_binary.py`, `main_binary_hinge.py`) uses `utils.ResultsLog`, which imports:
- `pandas`
- `bokeh`

If you run only the thesis experiment scripts, you can ignore these extra dependencies.

---

## 12) Known pitfalls / legacy details (important for reproducibility)

### 12.1 LogSoftmax vs CrossEntropyLoss in some legacy models
Some baseline models (and the baseline `main_mnist.py`) use `LogSoftmax` in the model forward, but also use `nn.CrossEntropyLoss()` as criterion in places. In PyTorch, `CrossEntropyLoss` expects **raw logits** (it applies `log_softmax` internally).  
Implication:
- If you evaluate legacy scripts, verify whether the model returns logits or log-probabilities.
- If it returns log-probabilities, use `nn.NLLLoss()` instead, or remove the final `LogSoftmax` from the model.

The thesis experiment scripts avoid this ambiguity by always computing on **raw logits** (`return_logits=True`) and applying losses explicitly.

### 12.2 Dataset paths in `data.py` are system-specific
`data.py` uses a hard-coded `_DATASETS_MAIN_PATH = '/home/Datasets'` (Linux-style). This is not portable to Windows without editing.  
The thesis experiment scripts are portable because they rely on `torchvision` downloads into `../data`.

### 12.3 `preprocess.py` has a non-returning helper
`scale_random_crop(...)` builds a transform pipeline but does not `return` it. If you use this helper, fix it or avoid it (the default paths mainly use `pad_random_crop` / `scale_crop`).

### 12.4 Debug breakpoints in `models/resnet_binary.py`
`Bottleneck.forward` contains a `pdb.set_trace()` breakpoint. If you try ImageNet ResNet depths that use `Bottleneck`, training will stop in the debugger unless removed.

---

## 13) Reproducibility checklist (recommended for thesis writing)

- **Seeds**: use `--seed` in experiment scripts; set `torch.manual_seed` (already done in experiments).  
- **Windows DataLoader**: pass `--num-workers 0` to avoid multiprocessing issues.  
- **Device**: explicitly note GPU/CPU used; runs on CPU will be very slow (especially CIFAR).  
- **Report saving**: keep both `experiments/results/*.txt` and `experiments/plots/*.png` for every figure you plan to include.

---

## 14) Thesis mapping (how this folder maps to chapters/appendix)

- **BNN background & STE training**: `models/binarized_modules.py`, plus baseline runners (`main_binary.py`).
- **Physics-inspired loss design**: `experiments/mnist_mcmc_experiment.py` (`VlogLoss`, schedulers, stability definition).
- **Scaling to vision tasks**: `experiments/cifar10_mcmc_experiment.py` (CIFAR-10) and the produced plots/results.
- **Result aggregation**: `experiments/summary_results/analyze_all_results.py` and `RESULTS_ANALYSIS.md/html`.