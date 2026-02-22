"""
Train CIFAR-10 Binary ResNet using existing repo modules:
- models/resnet_binary.py (resnet_binary factory)
- models/binarized_modules.py (BinarizeConv2d/BinarizeLinear + .org shadow weights)

This script keeps your repo minimally changed: it's just a new entrypoint.

Run:
  python experiments/train_cifar10_resnet_bnn.py --epochs 250 --batch-size 128 --lr 0.005

Optional:
  --loss ce|hinge|vtau_fixed|vtau_anneal
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

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
    V_tau potential (Straziota-style shape) applied to multiclass margin.

    s = (correct_logit - max_wrong_logit)/sqrt(C)
    U_tau(s) = (1/tau)(1 - s^tau)   for s>0
             = (1/tau)(1 - s)       for s<=0
    """
    def __init__(self, tau=1.0, num_classes=10):
        super().__init__()
        self.tau = float(tau)
        self.sqrt_c = (num_classes ** 0.5)

    def forward(self, logits, target):
        bs = logits.size(0)
        dev = logits.device

        correct = logits[torch.arange(bs, device=dev), target]

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(bs, device=dev), target] = False
        max_wrong = logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]

        s = (correct - max_wrong) / self.sqrt_c

        # numerical safety
        tau = max(self.tau, 1e-3)
        eps = 1e-6

        out = torch.zeros_like(s)
        pos = s > 0
        if pos.any():
            sp = torch.clamp(s[pos], min=eps)
            out[pos] = (1.0 - torch.pow(sp, tau)) / tau
        if (~pos).any():
            out[~pos] = (1.0 - s[~pos]) / tau
        return out.mean()


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


import torch

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
    p.add_argument("--tau", type=float, default=0.5)          # vtau_fixed
    p.add_argument("--tau-start", type=float, default=1.0)    # vtau_anneal
    p.add_argument("--tau-end", type=float, default=0.01)     # vtau_anneal

    args = p.parse_args()

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
        criterion = VtauLoss(tau=args.tau, num_classes=10)
        loss_desc = f"VtauFixed(tau={args.tau})"
        loss_type = "vtau_fixed"
    else:  # vtau_anneal
        criterion = VtauLoss(tau=args.tau_start, num_classes=10)
        tau_sched = TauScheduler(args.tau_start, args.tau_end, args.epochs)
        loss_desc = f"VtauAnneal({args.tau_start}->{args.tau_end})"
        loss_type = "vtau_anneal"

    # Optimizer: match your ResNet_cifar10.regime default idea (Adam)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 70)
    print(f"Model: resnet_binary(cifar10, depth={args.depth})")
    print(f"Loss:  {loss_desc}")
    print(f"Opt:   Adam(lr={args.lr}, wd={args.weight_decay})")
    print(f"Epochs:{args.epochs} | Batch:{args.batch_size} | Clip:{args.clip_norm}")
    print("=" * 70)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Apply tau schedule if needed
        if tau_sched is not None:
            criterion.tau = tau_sched.get_tau(epoch - 1)

        tr_l, tr_a = train_epoch(model, device, train_loader, optimizer, criterion, loss_type, args.clip_norm)
        te_l, te_a = eval_epoch(model, device, test_loader, criterion, loss_type)

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
    print("=" * 70)
    summarize_bnn_weights(model)
    print(f"Done in {elapsed/60:.1f} min | Best test acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()