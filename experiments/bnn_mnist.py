"""
MNIST Binary Neural Network: V_tau Loss Experiment.

Trains a BNN (Courbariaux et al., 2016) on MNIST comparing loss functions:
  1. ce:                 Cross-Entropy (baseline)
  2. hinge:              Hinge loss (baseline)
  3. vlog_fixed:         V_tau potential with fixed tau
  4. vlog_tau_annealing: V_tau potential with tau-annealing

V_tau potential (Straziota et al., Eq. 20):
  U_tau(s) = (1/tau)(1 - s^tau)   for s > 0
  U_tau(s) = (1/tau)(1 - s)       for s <= 0

Loss = (1/T) * U_tau(s), where T is the temperature (Straziota Eq. 12).
As tau -> 0, this approaches -log(s)/T. Straziota found T < 1 (e.g. 0.5)
is needed for the log-potential to suppress the spurious q=1 peak.

Tau-annealing starts with tau=1 (linear potential, easy to optimize) and
decreases toward tau~0 (log-potential, focuses on margin quality).

Architecture follows Courbariaux et al. (2016) MNIST MLP:
  BinarizeLinear -> BatchNorm -> Hardtanh (STE gradient clipping, Eq. 4)
  3 hidden layers, Dropout regularization, ADAM optimizer.
  Last layer is full-precision (standard BNN practice).

Usage:
  python experiments/bnn_mnist.py --loss-type ce --epochs 40 --lr 0.01
  python experiments/bnn_mnist.py --loss-type hinge --epochs 40 --lr 0.01
  python experiments/bnn_mnist.py --loss-type vlog_fixed --epochs 40 --lr 0.01 --tau-value 0.5 --temperature 0.5
  python experiments/bnn_mnist.py --loss-type vlog_tau_annealing --epochs 40 --lr 0.01 --tau-start 1.0 --tau-end 0.01 --temperature 0.5
"""

from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.binarized_modules import BinarizeLinear


# ============================================================================
# Loss Functions
# ============================================================================

class HingeLoss(nn.Module):
    """Multi-class hinge loss with {-1,+1} one-hot targets."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits, target_onehot):
        return torch.clamp(self.margin - logits * target_onehot, min=0).mean()


class VtauLoss(nn.Module):
    """
    V_tau potential loss for binary networks (Straziota et al., Eq. 20).

    U_tau(s) = (1/tau)(1 - s^tau)   for s > 0
    U_tau(s) = (1/tau)(1 - s)       for s <= 0

    The loss per sample is (1/T) * U_tau(s), where T is the temperature
    (Straziota Eq. 12). Stability s = (correct - max_wrong) / sqrt(C).

    The 1/tau prefactor ensures the gradient for s > 0 is -s^{tau-1},
    which smoothly approaches -1/s (gradient of -log s) as tau -> 0.
    Straziota found T < 1 (e.g. T=0.5) is needed for the log-potential
    to suppress the spurious peak at q=1 in the binary perceptron.
    """

    def __init__(self, tau=1.0, temperature=1.0, num_classes=10):
        super().__init__()
        self.tau = tau
        self.temperature = temperature
        self.sqrt_n = np.sqrt(num_classes)

    def forward(self, logits, target):
        bs = logits.size(0)
        dev = logits.device

        correct = logits[torch.arange(bs, device=dev), target]

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(bs, device=dev), target] = False
        max_wrong = logits.masked_fill(~mask, float('-inf')).max(dim=1)[0]

        s = (correct - max_wrong) / self.sqrt_n

        tau = max(self.tau, 1e-3)
        eps = 1e-6
        pos = s > 0

        result = torch.zeros_like(s)
        if pos.any():
            sp = torch.clamp(s[pos], min=eps)
            result[pos] = (1.0 - torch.pow(sp, tau)) / tau
        if (~pos).any():
            result[~pos] = (1.0 - s[~pos]) / tau

        return result.mean() / max(self.temperature, 1e-6)


class TauScheduler:
    """Linear annealing: tau decreases from tau_start to tau_end."""

    def __init__(self, tau_start, tau_end, total_epochs):
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.denom = max(total_epochs - 1, 1)

    def get_tau(self, epoch):
        t = min(epoch / self.denom, 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * t


# ============================================================================
# Network Architecture
# ============================================================================

class BinaryNet(nn.Module):
    """
    3-hidden-layer BNN for MNIST (Courbariaux et al., 2016).

    Each hidden layer: BinarizeLinear -> BatchNorm -> Hardtanh.
    Hardtanh implements the STE gradient clipping from Eq. 4 of the paper.
    Dropout provides regularization (used in the Theano version of the paper).
    Last layer is full-precision Linear -> raw logits.
    """

    def __init__(self, hidden=2048, inflate=3):
        super().__init__()
        h = hidden * inflate
        self.fc1 = BinarizeLinear(784, h)
        self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = BinarizeLinear(h, h)
        self.bn2 = nn.BatchNorm1d(h)
        self.fc3 = BinarizeLinear(h, h)
        self.bn3 = nn.BatchNorm1d(h)
        self.fc4 = nn.Linear(h, 10)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.nn.functional.hardtanh(self.bn1(self.fc1(x)))
        x = torch.nn.functional.hardtanh(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = torch.nn.functional.hardtanh(self.bn3(self.fc3(x)))
        return self.fc4(x)


# ============================================================================
# Training and Testing
# ============================================================================

def train_epoch(model, device, loader, optimizer, criterion, loss_type,
                clip_norm):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        if loss_type == 'hinge':
            target_oh = -torch.ones(target.size(0), 10, device=device)
            target_oh.scatter_(1, target.unsqueeze(1), 1)
            loss = criterion(output, target_oh)
        else:
            loss = criterion(output, target)

        loss.backward()

        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        for p in model.parameters():
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in model.parameters():
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))

        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, 100.0 * correct / total


def test_epoch(model, device, loader, criterion, loss_type):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if loss_type == 'hinge':
                target_oh = -torch.ones(target.size(0), 10, device=device)
                target_oh.scatter_(1, target.unsqueeze(1), 1)
                total_loss += criterion(output, target_oh).item() * data.size(0)
            else:
                total_loss += criterion(output, target).item() * data.size(0)

            correct += output.argmax(1).eq(target).sum().item()
            total += data.size(0)

    return total_loss / total, 100.0 * correct / total


# ============================================================================
# Plotting
# ============================================================================

def save_plot(tr_l, te_l, tr_a, te_a, name, save_dir, title):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(tr_l) + 1)

    ax1.plot(ep, tr_l, 'b-', label='Train', linewidth=2)
    ax1.plot(ep, te_l, 'r-', label='Test', linewidth=2)
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep, tr_a, 'b-', label='Train', linewidth=2)
    ax2.plot(ep, te_a, 'r-', label='Test', linewidth=2)
    ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description='MNIST BNN: V_tau Loss Experiment')
    p.add_argument('--batch-size', type=int, default=100,
                   help='Training batch size (default: 100, matches Courbariaux)')
    p.add_argument('--test-batch-size', type=int, default=1000)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--lr-decay-step', type=int, default=0,
                   help='Decay LR every N epochs (0=disabled)')
    p.add_argument('--lr-decay-factor', type=float, default=0.1)
    p.add_argument('--no-cuda', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--clip-norm', type=float, default=1.0,
                   help='Gradient clip norm (0=disable)')

    p.add_argument('--loss-type', type=str, default='ce',
                   choices=['ce', 'hinge', 'vlog_fixed', 'vlog_tau_annealing'])
    p.add_argument('--hinge-margin', type=float, default=1.0)
    p.add_argument('--tau-value', type=float, default=0.5,
                   help='Fixed tau for vlog_fixed')
    p.add_argument('--tau-start', type=float, default=1.0,
                   help='Starting tau for annealing')
    p.add_argument('--tau-end', type=float, default=0.01,
                   help='Final tau for annealing')
    p.add_argument('--temperature', type=float, default=1.0,
                   help='Temperature T in exp(-U/T) (Straziota Eq. 12, try 0.5)')

    p.add_argument('--plot-dir', type=str, default='experiments/plots')
    p.add_argument('--no-plot', action='store_true', default=False)
    args = p.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kw = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU mode (training will be slow)")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.batch_size, shuffle=True, **kw)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=args.test_batch_size, shuffle=False, **kw)

    model = BinaryNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    tau_sched = None
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
        desc = 'Cross-Entropy'
    elif args.loss_type == 'hinge':
        criterion = HingeLoss(margin=args.hinge_margin)
        desc = f'Hinge (margin={args.hinge_margin})'
    elif args.loss_type == 'vlog_fixed':
        criterion = VtauLoss(tau=args.tau_value, temperature=args.temperature)
        desc = f'Vtau fixed (tau={args.tau_value}, T={args.temperature})'
    elif args.loss_type == 'vlog_tau_annealing':
        criterion = VtauLoss(tau=args.tau_start, temperature=args.temperature)
        tau_sched = TauScheduler(args.tau_start, args.tau_end, args.epochs)
        desc = f'Vtau anneal (tau {args.tau_start}->{args.tau_end}, T={args.temperature})'

    print(f"Loss: {desc}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Gradient clip norm: {args.clip_norm}")
    print('=' * 70)

    tr_losses, te_losses, tr_accs, te_accs = [], [], [], []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        if args.lr_decay_step > 0 and epoch > 1 and (epoch - 1) % args.lr_decay_step == 0:
            for pg in optimizer.param_groups:
                pg['lr'] *= args.lr_decay_factor
            print(f"  LR decayed to {optimizer.param_groups[0]['lr']:.6f}")

        if tau_sched is not None:
            criterion.tau = tau_sched.get_tau(epoch - 1)

        tr_l, tr_a = train_epoch(model, device, train_loader, optimizer,
                                 criterion, args.loss_type, args.clip_norm)
        te_l, te_a = test_epoch(model, device, test_loader,
                                criterion, args.loss_type)

        tr_losses.append(tr_l)
        te_losses.append(te_l)
        tr_accs.append(tr_a)
        te_accs.append(te_a)

        extra = ''
        if hasattr(criterion, 'tau'):
            extra = f' tau={criterion.tau:.4f}'
        print(f'Ep {epoch:3d}{extra} | '
              f'Train: loss={tr_l:.4f} acc={tr_a:.1f}% | '
              f'Test: loss={te_l:.4f} acc={te_a:.1f}%')

    elapsed = time.time() - t0
    best_acc = max(te_accs)
    best_ep = te_accs.index(best_acc) + 1
    print('=' * 70)
    print(f'Done in {elapsed:.0f}s ({elapsed/60:.1f}min), '
          f'{elapsed/args.epochs:.1f}s/epoch')
    print(f'Best test accuracy: {best_acc:.2f}% (epoch {best_ep})')

    name = f'mnist_{args.loss_type}'
    if args.loss_type == 'hinge':
        name += f'_m{args.hinge_margin}'
    elif args.loss_type == 'vlog_fixed':
        name += f'_tau{args.tau_value}'
    elif args.loss_type == 'vlog_tau_annealing':
        name += f'_tau{args.tau_start}-{args.tau_end}'
    name += f'_e{args.epochs}_lr{args.lr}'

    title = f'{desc} | LR={args.lr} Batch={args.batch_size}'

    if not args.no_plot:
        save_plot(tr_losses, te_losses, tr_accs, te_accs,
                  name, args.plot_dir, title)

    base = os.path.dirname(args.plot_dir) if args.plot_dir.endswith('plots') else args.plot_dir
    rdir = os.path.join(base, 'results')
    os.makedirs(rdir, exist_ok=True)
    rpath = os.path.join(rdir, f'{name}.txt')
    with open(rpath, 'w') as f:
        f.write(f'Experiment: {name}\n{"="*60}\n')
        f.write(f'Loss: {desc}\n')
        f.write(f'Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}\n')
        f.write(f'Clip norm: {args.clip_norm}\n')
        if args.lr_decay_step > 0:
            f.write(f'LR decay: every {args.lr_decay_step} epochs, '
                    f'factor={args.lr_decay_factor}\n')
        f.write(f'Time: {elapsed:.0f}s, '
                f'Best test: {best_acc:.2f}% (epoch {best_ep})\n\n')
        f.write(f'{"Ep":<5} {"TrLoss":<12} {"TeLoss":<12} '
                f'{"TrAcc%":<10} {"TeAcc%":<10}\n')
        f.write('-' * 50 + '\n')
        for i in range(len(te_accs)):
            f.write(f'{i+1:<5} {tr_losses[i]:<12.6f} {te_losses[i]:<12.6f} '
                    f'{tr_accs[i]:<10.2f} {te_accs[i]:<10.2f}\n')
    print(f'Results saved: {rpath}')


if __name__ == '__main__':
    main()