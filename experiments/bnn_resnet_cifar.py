"""
CIFAR-10 Binary ResNet: V_tau Loss Experiment.

Trains a binary ResNet (Courbariaux et al., 2016) on CIFAR-10 comparing:
  1. ce:          Cross-Entropy (baseline)
  2. hinge:       Hinge loss (baseline)
  3. vtau_fixed:  V_tau potential with fixed tau
  4. vtau_anneal: V_tau potential with tau-annealing

V_tau potential (Straziota et al., Eq. 20):
  U_tau(s) = (1/tau)(1 - s^tau)   for s > 0
  U_tau(s) = (1/tau)(1 - s)       for s <= 0

Loss = (1/T) * U_tau(s), where T is the temperature (Eq. 12).
As tau -> 0, this approaches -log(s)/T.

Architecture: models/resnet_binary.py (BinarizeConv2d, .org shadow weights).

Usage:
  python experiments/bnn_resnet_cifar.py --loss ce --epochs 250 --lr 0.005
  python experiments/bnn_resnet_cifar.py --loss vtau_anneal --epochs 250 --lr 0.005 --temperature 0.5
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Adjust imports to match your repo layout:
# If this script is in experiments/, and models/ is sibling:
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.resnet_binary import resnet_binary  # uses your existing resnet_binary.py


# =========================
# Losses (optional additions)
# =========================

class HingeLoss(nn.Module):
    """Multi-class hinge loss with {-1,+1} one-hot targets."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits, target_onehot):
        return torch.clamp(self.margin - logits * target_onehot, min=0).mean()


class VtauLoss(nn.Module):
    """
    V_tau potential (Straziota et al., Eq. 20) applied to multiclass margin.

    s = (correct_logit - max_wrong_logit)/sqrt(C)
    U_tau(s) = (1/tau)(1 - s^tau)   for s>0
             = (1/tau)(1 - s)       for s<=0

    Loss = (1/T) * U_tau(s), with temperature T (Straziota Eq. 12).
    T < 1 (e.g. T=0.5) suppresses the spurious q=1 peak in the binary
    perceptron free entropy, enabling efficient sampling via ASL.
    """
    def __init__(self, tau=1.0, temperature=1.0, num_classes=10):
        super().__init__()
        self.tau = float(tau)
        self.temperature = float(temperature)
        self.sqrt_c = (num_classes ** 0.5)

    def forward(self, logits, target):
        bs = logits.size(0)
        dev = logits.device

        correct = logits[torch.arange(bs, device=dev), target]

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(bs, device=dev), target] = False
        max_wrong = logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]

        s = (correct - max_wrong) / self.sqrt_c

        tau = max(self.tau, 1e-3)
        eps = 1e-6

        out = torch.zeros_like(s)
        pos = s > 0
        if pos.any():
            sp = torch.clamp(s[pos], min=eps)
            out[pos] = (1.0 - torch.pow(sp, tau)) / tau
        if (~pos).any():
            out[~pos] = (1.0 - s[~pos]) / tau
        return out.mean() / max(self.temperature, 1e-6)


class TauScheduler:
    """Linear annealing: tau decreases from tau_start to tau_end across epochs."""
    def __init__(self, tau_start, tau_end, total_epochs):
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.denom = max(total_epochs - 1, 1)

    def get_tau(self, epoch_idx_zero_based: int) -> float:
        t = min(epoch_idx_zero_based / self.denom, 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * t


# =========================
# Shadow-weight handling (matches your repo pattern)
# =========================

def restore_org_params_(model: nn.Module):
    """Before optimizer.step(): copy p.org -> p.data for binarized params."""
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, "org"):
                p.data.copy_(p.org)

def clamp_and_store_org_params_(model: nn.Module, lo=-1.0, hi=1.0):
    """After optimizer.step(): clamp p.data and store into p.org."""
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, "org"):
                p.data.clamp_(lo, hi)
                p.org.copy_(p.data)


# =========================
# Train / eval loops
# =========================

def train_epoch(model, device, loader, optimizer, criterion, loss_type, clip_norm):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        if loss_type == "hinge":
            y_oh = -torch.ones(y.size(0), 10, device=device)
            y_oh.scatter_(1, y.unsqueeze(1), 1)
            loss = criterion(logits, y_oh)
        else:
            loss = criterion(logits, y)

        loss.backward()

        if clip_norm and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        # IMPORTANT: follow your repo’s BNN update protocol
        restore_org_params_(model)
        optimizer.step()
        clamp_and_store_org_params_(model)

        total_loss += loss.item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, device, loader, criterion, loss_type):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        if loss_type == "hinge":
            y_oh = -torch.ones(y.size(0), 10, device=device)
            y_oh.scatter_(1, y.unsqueeze(1), 1)
            loss = criterion(logits, y_oh)
        else:
            loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += logits.argmax(1).eq(y).sum().item()
        total += x.size(0)

    return total_loss / total, 100.0 * correct / total


def summarize_bnn_weights(model, max_print=8):
    """
    Prints summary stats for parameters with shadow copy (.org),
    plus a small sample of real and binarized values.
    """
    print("\n=== Weight summary (real vs binarized) ===")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Only binarized layers create .org in your repo
        has_org = hasattr(p, "org")
        p_real = p.org if has_org else p.data

        # Compute binarized view (as used in forward)
        p_bin = p_real.sign()

        # Stats
        real_min = p_real.min().item()
        real_max = p_real.max().item()
        real_mean = p_real.mean().item()
        bin_unique = torch.unique(p_bin).detach().cpu().tolist()

        # Small sample
        flat_real = p_real.view(-1)[:max_print].detach().cpu().tolist()
        flat_bin = p_bin.view(-1)[:max_print].detach().cpu().tolist()

        tag = "BINARIZED_PARAM" if has_org else "FP_PARAM"
        print(f"\n{name} [{tag}] shape={tuple(p.shape)}")
        print(f"  real: min={real_min:.3f} max={real_max:.3f} mean={real_mean:.3f}")
        print(f"  bin unique values: {bin_unique}")
        print(f"  sample real: {flat_real}")
        print(f"  sample bin : {flat_bin}")


# =========================
# Plotting
# =========================

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


# =========================
# Main entrypoint
# =========================

def main():
    p = argparse.ArgumentParser("CIFAR-10 Binary ResNet (minimal repo changes)")
    p.add_argument("--data-dir", type=str, default="../data")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--clip-norm", type=float, default=1.0)

    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--loss", type=str, default="ce",
                   choices=["ce", "hinge", "vtau_fixed", "vtau_anneal"])
    p.add_argument("--hinge-margin", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature T in exp(-U/T) (Straziota Eq. 12, try 0.5)")

    p.add_argument("--lr-decay-epochs", type=str, default="101,142,184,220",
                   help="Comma-separated epochs to decay LR (matches ResNet_cifar10 regime)")
    p.add_argument("--lr-decay-factor", type=float, default=0.2)

    p.add_argument("--plot-dir", type=str, default="experiments/plots")
    p.add_argument("--no-plot", action="store_true", default=False)

    args = p.parse_args()
    args.lr_decay_epochs = [int(e) for e in args.lr_decay_epochs.split(",")]

    torch.manual_seed(args.seed)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("Device: CUDA")
    else:
        print("Device: CPU")

    # CIFAR-10 transforms (standard augmentation)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    te_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    kw = {"num_workers": args.num_workers, "pin_memory": True} if use_cuda else {"num_workers": args.num_workers}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tr_tf),
        batch_size=args.batch_size, shuffle=True, **kw
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(args.data_dir, train=False, download=True, transform=te_tf),
        batch_size=args.test_batch_size, shuffle=False, **kw
    )

    # Instantiate your existing binary ResNet
    model = resnet_binary(dataset="cifar10", depth=args.depth, num_classes=10).to(device)

    # Loss selection
    tau_sched = None
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
        loss_desc = "CrossEntropy"
        loss_type = "ce"
    elif args.loss == "hinge":
        criterion = HingeLoss(margin=args.hinge_margin)
        loss_desc = f"Hinge(m={args.hinge_margin})"
        loss_type = "hinge"
    elif args.loss == "vtau_fixed":
        criterion = VtauLoss(tau=args.tau, temperature=args.temperature, num_classes=10)
        loss_desc = f"VtauFixed(tau={args.tau}, T={args.temperature})"
        loss_type = "vtau_fixed"
    else:  # vtau_anneal
        criterion = VtauLoss(tau=args.tau_start, temperature=args.temperature, num_classes=10)
        tau_sched = TauScheduler(args.tau_start, args.tau_end, args.epochs)
        loss_desc = f"VtauAnneal({args.tau_start}->{args.tau_end}, T={args.temperature})"
        loss_type = "vtau_anneal"

    # Optimizer: match your ResNet_cifar10.regime default idea (Adam)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 70)
    print(f"Model: resnet_binary(cifar10, depth={args.depth})")
    print(f"Loss:  {loss_desc}")
    print(f"Opt:   Adam(lr={args.lr}, wd={args.weight_decay})")
    print(f"Epochs:{args.epochs} | Batch:{args.batch_size} | Clip:{args.clip_norm}")
    print(f"LR decay at epochs {args.lr_decay_epochs}, factor={args.lr_decay_factor}")
    print("=" * 70)

    tr_losses, te_losses, tr_accs, te_accs = [], [], [], []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch in args.lr_decay_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] *= args.lr_decay_factor
            print(f"  LR decayed to {optimizer.param_groups[0]['lr']:.6f}")

        if tau_sched is not None:
            criterion.tau = tau_sched.get_tau(epoch - 1)

        tr_l, tr_a = train_epoch(model, device, train_loader, optimizer,
                                 criterion, loss_type, args.clip_norm)
        te_l, te_a = eval_epoch(model, device, test_loader,
                                criterion, loss_type)

        tr_losses.append(tr_l)
        te_losses.append(te_l)
        tr_accs.append(tr_a)
        te_accs.append(te_a)

        if te_a > best_acc:
            best_acc = te_a

        extra = ""
        if hasattr(criterion, "tau"):
            extra = f" tau={criterion.tau:.4f}"

        print(
            f"Ep {epoch:3d}{extra} | "
            f"Train: loss={tr_l:.4f} acc={tr_a:.2f}% | "
            f"Test: loss={te_l:.4f} acc={te_a:.2f}% | "
            f"Best={best_acc:.2f}%"
        )

    elapsed = time.time() - t0
    best_ep = te_accs.index(best_acc) + 1
    print("=" * 70)
    summarize_bnn_weights(model)
    print(f"Done in {elapsed/60:.1f} min | Best test acc: {best_acc:.2f}% (epoch {best_ep})")

    name = f"cifar10_{args.loss}_d{args.depth}"
    if args.loss == "hinge":
        name += f"_m{args.hinge_margin}"
    elif args.loss == "vtau_fixed":
        name += f"_tau{args.tau}_T{args.temperature}"
    elif args.loss == "vtau_anneal":
        name += f"_tau{args.tau_start}-{args.tau_end}_T{args.temperature}"
    name += f"_e{args.epochs}_lr{args.lr}"

    title = f"{loss_desc} | LR={args.lr} Batch={args.batch_size}"

    if not args.no_plot:
        save_plot(tr_losses, te_losses, tr_accs, te_accs,
                  name, args.plot_dir, title)

    base = os.path.dirname(args.plot_dir) if args.plot_dir.endswith("plots") else args.plot_dir
    rdir = os.path.join(base, "results")
    os.makedirs(rdir, exist_ok=True)
    rpath = os.path.join(rdir, f"{name}.txt")
    with open(rpath, "w") as f:
        f.write(f"Experiment: {name}\n{'='*60}\n")
        f.write(f"Loss: {loss_desc}\n")
        f.write(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}\n")
        f.write(f"Clip norm: {args.clip_norm}\n")
        f.write(f"LR decay at {args.lr_decay_epochs}, factor={args.lr_decay_factor}\n")
        f.write(f"Time: {elapsed:.0f}s, Best test: {best_acc:.2f}% (epoch {best_ep})\n\n")
        f.write(f"{'Ep':<5} {'TrLoss':<12} {'TeLoss':<12} "
                f"{'TrAcc%':<10} {'TeAcc%':<10}\n")
        f.write("-" * 50 + "\n")
        for i in range(len(te_accs)):
            f.write(f"{i+1:<5} {tr_losses[i]:<12.6f} {te_losses[i]:<12.6f} "
                    f"{tr_accs[i]:<10.2f} {te_accs[i]:<10.2f}\n")
    print(f"Results saved: {rpath}")

if __name__ == "__main__":
    main()