"""
MNIST Binary Network with MCMC-inspired Loss Functions (minimal).
Compares: CE, Hinge (baseline), Vlog (fixed b & beta), Vlog (fixed b=1, beta-annealing).

Example commands (always set --lr; vlog_fixed uses --b-value and --beta-fixed; vlog_annealing uses --beta-start and --beta-end):

  python experiments/mnist_mcmc_experiment.py --loss-type ce --epochs 40 --batch-size 64 --lr 0.01 --plot-dir experiments/plots

  python experiments/mnist_mcmc_experiment.py --loss-type hinge --epochs 40 --batch-size 64 --lr 0.01 --hinge-margin 1.0 --plot-dir experiments/plots

  python experiments/mnist_mcmc_experiment.py --loss-type vlog_fixed --epochs 40 --batch-size 64 --lr 0.01 --b-value 10 --beta-fixed 1.0 --plot-dir experiments/plots

  python experiments/mnist_mcmc_experiment.py --loss-type vlog_annealing --epochs 40 --batch-size 64 --lr 0.01 --beta-start 0.5 --beta-end 5.0 --plot-dir experiments/plots
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.binarized_modules import BinarizeLinear, Binarize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures


# ============================================================================
# Loss Functions
# ============================================================================

class HingeLoss(nn.Module):
    """Standard Hinge (SVM-style): L = mean(max(0, margin - y * f(x))). One-hot targets: -1/+1."""
    def __init__(self, margin=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target_onehot):
        # input: (batch_size, num_classes), target_onehot: (batch_size, num_classes) with -1/+1
        output = self.margin - input.mul(target_onehot)
        output = torch.clamp(output, min=0)
        return output.mean()


# ============================================================================
# MCMC-Inspired Loss Functions
# ============================================================================

class VlogLoss(nn.Module):
    """
    Vlog potential: V(x,b) = 1 - x^(1/b) for x>0, 1-x for x<=0.
    Loss = beta * mean(V(stabilities, b)); stability = (correct - max_wrong) / sqrt(N).
    """
    def __init__(self, b=1.0, beta=1.0, normalization_dim=10):
        super(VlogLoss, self).__init__()
        self.b = b
        self.beta = beta
        self.sqrt_n = np.sqrt(normalization_dim)

    def vlog_potential(self, x):
        result = torch.zeros_like(x)
        pos = x > 0
        if pos.any():
            result[pos] = 1.0 - torch.clamp(torch.pow(x[pos], 1.0 / self.b), max=10.0)
        result[~pos] = 1.0 - x[~pos]
        return result

    def compute_stabilities(self, output, target):
        batch_size, num_classes = output.size(0), output.size(1)
        correct_scores = output[torch.arange(batch_size), target]
        mask = torch.ones_like(output).bool()
        mask[torch.arange(batch_size), target] = False
        max_wrong_scores = output.masked_fill(~mask, float('-inf')).max(dim=1)[0]
        return (correct_scores - max_wrong_scores) / self.sqrt_n

    def forward(self, output, target):
        stabilities = self.compute_stabilities(output, target)
        return self.beta * self.vlog_potential(stabilities).mean()

    def update_beta(self, new_beta):
        self.beta = new_beta


class BetaScheduler:
    """
    Scheduler for beta-annealing: beta increases from beta_start to beta_end
    As beta increases, we focus more on minimizing the loss (simulated annealing)
    """
    def __init__(self, beta_start=0.1, beta_end=100.0, total_epochs=100, schedule_type='linear'):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
    
    def get_beta(self, epoch):
        """Get beta value for current epoch"""
        if self.schedule_type == 'linear':
            # Linear interpolation
            alpha = epoch / self.total_epochs
            beta = self.beta_start + (self.beta_end - self.beta_start) * alpha
        elif self.schedule_type == 'exponential':
            # Exponential growth
            alpha = epoch / self.total_epochs
            beta = self.beta_start * (self.beta_end / self.beta_start) ** alpha
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return beta


# ============================================================================
# Network Architecture
# ============================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 3
        self.fc1 = BinarizeLinear(784, 2048*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc2 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc3 = BinarizeLinear(2048*self.infl_ratio, 2048*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048*self.infl_ratio)
        self.fc4 = nn.Linear(2048*self.infl_ratio, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1) 
        self.drop = nn.Dropout(0.5)

    def forward(self, x, return_logits=False):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        
        if return_logits:
            return x  # Return raw logits for Vlog loss
        return self.logsoftmax(x)


# ============================================================================
# Plotting and Visualization
# ============================================================================

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, 
                         experiment_name, save_dir='experiments/plots', args=None):
    """
    Plot and save training curves (loss and accuracy)
    
    Args:
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
        train_accs: List of training accuracies per epoch
        test_accs: List of test accuracies per epoch
        experiment_name: Name for the plot file
        save_dir: Directory to save plots
        args: Arguments dict for adding experiment details to title
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Epochs', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([80, 100])
    
    if args is not None:
        if args.loss_type == 'ce':
            title = f'Cross-Entropy | LR={args.lr} Batch={args.batch_size}'
        elif args.loss_type == 'hinge':
            title = f'Hinge (margin={args.hinge_margin}) | LR={args.lr} Batch={args.batch_size}'
        elif args.loss_type == 'vlog_fixed':
            title = f'Vlog fixed (b={args.b_value}, β={args.beta_fixed}) | LR={args.lr} Batch={args.batch_size}'
        elif args.loss_type == 'vlog_annealing':
            title = f'Vlog β-anneal ({args.beta_start}→{args.beta_end}, b=1) | LR={args.lr} Batch={args.batch_size}'
        else:
            title = experiment_name
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'{experiment_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {save_path}")
    plt.close()
    
    return save_path


# ============================================================================
# Training and Testing
# ============================================================================

def train(model, device, train_loader, optimizer, criterion, epoch, args, beta_scheduler=None):
    model.train()
    total_loss = 0
    correct = 0

    if beta_scheduler is not None:
        current_beta = beta_scheduler.get_beta(epoch - 1)
        criterion.update_beta(current_beta)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: Beta = {current_beta:.4f}')

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Always use raw logits (CE loss expects them, Vlog/Hinge also use them)
        output = model(data, return_logits=True)
        
        # HingeLoss requires one-hot encoding with {-1, +1}
        if isinstance(criterion, HingeLoss):
            num_classes = output.size(1)
            target_onehot = torch.zeros(target.size(0), num_classes, device=device)
            target_onehot.fill_(-1)
            target_onehot.scatter_(1, target.unsqueeze(1), 1)
            loss = criterion(output, target_onehot)
        else:
            loss = criterion(output, target)
        loss.backward()
        
        # Binary network weight update logic
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))
        
        total_loss += loss.item()
        
        # Calculate accuracy (same for all losses)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    return avg_loss, accuracy


def test(model, device, test_loader, criterion, args):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Always use raw logits (CE loss expects them, Vlog/Hinge also use them)
            output = model(data, return_logits=True)
            
            # HingeLoss requires one-hot encoding with {-1, +1}
            if isinstance(criterion, HingeLoss):
                num_classes = output.size(1)
                target_onehot = torch.zeros(target.size(0), num_classes, device=device)
                target_onehot.fill_(-1)
                target_onehot.scatter_(1, target.unsqueeze(1), 1)
                test_loss += criterion(output, target_onehot).item()
            else:
                test_loss += criterion(output, target).item()
            
            # Get predictions (same for all losses)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    # Divide by number of batches (same as training) for comparable scale
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MNIST BinaryNet with MCMC Loss')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-decay-step', type=int, default=40, metavar='N',
                        help='decay learning rate every N epochs (default: 40, set to 0 to disable)')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1, metavar='F',
                        help='multiply learning rate by this factor at each decay step (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of data loading workers (default: 4, use 0 for Windows)')
    
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'hinge', 'vlog_fixed', 'vlog_annealing'],
                        help='ce | hinge | vlog_fixed (use --b-value, --beta-fixed) | vlog_annealing (use --beta-start, --beta-end)')
    parser.add_argument('--hinge-margin', type=float, default=1.0, help='Hinge margin (default: 1.0)')
    parser.add_argument('--b-value', type=float, default=10.0, help='Vlog fixed b for vlog_fixed only (default: 10.0)')
    parser.add_argument('--beta-fixed', type=float, default=1.0, help='Vlog set beta for vlog_fixed only (default: 1.0)')
    parser.add_argument('--beta-start', type=float, default=0.5, help='Vlog beta-annealing start for vlog_annealing only (default: 0.5)')
    parser.add_argument('--beta-end', type=float, default=5.0, help='Vlog beta-annealing end for vlog_annealing only (default: 5.0)')
    parser.add_argument('--normalization-dim', type=int, default=10, help='Stability norm dim (default: 10)')
    
    # Plotting and saving options
    parser.add_argument('--plot-dir', type=str, default='experiments/plots',
                        help='Directory to save plots (default: experiments/plots)')
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    # Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Data loading optimization
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    if use_cuda:
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Data workers: {args.num_workers}")
    else:
        print("⚠️ Using CPU (training will be slow!)")
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    # Model
    model = Net().to(device)
    
    beta_scheduler = None
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
        print("Using Cross-Entropy Loss")
    elif args.loss_type == 'hinge':
        criterion = HingeLoss(margin=args.hinge_margin).to(device)
        print(f"Using Hinge Loss (margin={args.hinge_margin})")
    elif args.loss_type == 'vlog_fixed':
        criterion = VlogLoss(b=args.b_value, beta=args.beta_fixed, normalization_dim=args.normalization_dim).to(device)
        print(f"Using Vlog Loss (fixed b={args.b_value}, beta={args.beta_fixed})")
    elif args.loss_type == 'vlog_annealing':
        criterion = VlogLoss(b=1.0, beta=args.beta_start, normalization_dim=args.normalization_dim).to(device)
        beta_scheduler = BetaScheduler(beta_start=args.beta_start, beta_end=args.beta_end,
                                       total_epochs=args.epochs, schedule_type='linear')
        print(f"Using Vlog Loss (b=1 fixed, beta {args.beta_start} -> {args.beta_end})")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    if args.lr_decay_step > 0:
        print(f"LR decay: every {args.lr_decay_step} epochs, multiply by {args.lr_decay_factor}")
    else:
        print("LR decay: disabled")
    print("="*70)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Start timing
    import time
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Learning rate decay (configurable via --lr-decay-step and --lr-decay-factor)
        if args.lr_decay_step > 0 and epoch % args.lr_decay_step == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_factor
            print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']}")
        
        train_loss, train_acc = train(model, device, train_loader, optimizer,
                                      criterion, epoch, args, beta_scheduler)
        test_loss, test_acc = test(model, device, test_loader, criterion, args)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    training_time_minutes = training_time / 60
    
    # Final summary
    print("="*70)
    print("Training Complete!")
    print(f"Training Time: {training_time:.1f}s ({training_time_minutes:.2f} min)")
    print(f"Time per Epoch: {training_time/args.epochs:.1f}s")
    print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"Best Test Accuracy: {max(test_accs):.2f}% (Epoch {test_accs.index(max(test_accs))+1})")
    
    experiment_name = f'mnist_{args.loss_type}'
    if args.loss_type == 'hinge':
        experiment_name += f'_m{args.hinge_margin}'
    elif args.loss_type == 'vlog_fixed':
        experiment_name += f'_b{args.b_value}_beta{args.beta_fixed}'
    elif args.loss_type == 'vlog_annealing':
        experiment_name += f'_beta{args.beta_start}-{args.beta_end}'
    experiment_name += f'_e{args.epochs}_bs{args.batch_size}_lr{args.lr}'
    
    # Plot training curves
    if not args.no_plot:
        plot_training_curves(train_losses, test_losses, train_accs, test_accs,
                           experiment_name, save_dir=args.plot_dir, args=args)
    
    # Save results to same base directory as plots
    # If plot_dir ends with 'plots', go up one level, otherwise use plot_dir's parent
    if args.plot_dir.endswith('plots'):
        base_dir = os.path.dirname(args.plot_dir)
    else:
        base_dir = args.plot_dir
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Use same naming convention as plots
    results_file = os.path.join(results_dir, f'{experiment_name}.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"="*60 + "\n\n")
        
        # Training configuration
        f.write(f"CONFIGURATION:\n")
        f.write(f"Loss Type: {args.loss_type}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        if args.lr_decay_step > 0:
            f.write(f"LR Decay: every {args.lr_decay_step} epochs, factor={args.lr_decay_factor}\n")
        else:
            f.write(f"LR Decay: disabled\n")
        f.write(f"Num Workers: {args.num_workers}\n")
        
        if args.loss_type == 'hinge':
            f.write(f"Hinge Margin: {args.hinge_margin}\n")
        elif args.loss_type == 'vlog_fixed':
            f.write(f"b: {args.b_value}, beta: {args.beta_fixed}\n")
        elif args.loss_type == 'vlog_annealing':
            f.write(f"b: 1 (fixed), beta: {args.beta_start} -> {args.beta_end} (linear)\n")
        
        # Training time
        f.write(f"\nTRAINING TIME:\n")
        f.write(f"Total Time: {training_time:.1f}s ({training_time_minutes:.2f} min)\n")
        f.write(f"Time per Epoch: {training_time/args.epochs:.1f}s\n")
        
        # Results
        f.write(f"\nRESULTS:\n")
        f.write(f"Final Test Accuracy: {test_accs[-1]:.2f}%\n")
        f.write(f"Best Test Accuracy: {max(test_accs):.2f}% (Epoch {test_accs.index(max(test_accs))+1})\n")
        
        # Detailed per-epoch results
        f.write(f"\nPER-EPOCH RESULTS:\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Test Loss':<15} {'Train Acc':<12} {'Test Acc':<12}\n")
        f.write(f"{'-'*70}\n")
        for i in range(len(test_accs)):
            f.write(f"{i+1:<8} {train_losses[i]:<15.6f} {test_losses[i]:<15.6f} {train_accs[i]:<12.2f}% {test_accs[i]:<12.2f}%\n")
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == '__main__':
    main()

