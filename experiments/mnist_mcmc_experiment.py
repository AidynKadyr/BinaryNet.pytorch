"""
MNIST Binary Network with MCMC-inspired Loss Functions
Compares:
1. Standard Cross-Entropy
2. HingeLoss (SVM-style margin-based)
3. Vlog potential (fixed beta, constant b)
4. Vlog potential with beta-annealing (beta: small -> large)
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
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
    """
    Parametric Hinge Loss with b-annealing and β-annealing support
    
    Standard Hinge: L = mean(max(0, margin - y * f(x)))
    With b parameter: L = mean(max(0, margin - y * f(x))^b)  [b controls sharpness]
    With β parameter: L = β * mean(...)  [β controls loss scale]
    
    b-annealing: b goes from 1 → large (sharpens the loss)
    β-annealing: β goes from small → large (increases loss magnitude)
    """
    def __init__(self, margin=1.0, b=1.0, beta=1.0):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.b = b  # Sharpness parameter (1 = standard hinge)
        self.beta = beta  # Temperature parameter (1 = standard scale)
    
    def forward(self, input, target_onehot):
        """
        Args:
            input: (batch_size, num_classes) - raw logits
            target_onehot: (batch_size, num_classes) - one-hot encoded with -1/+1
        """
        # Hinge loss: max(0, margin - y * f(x))
        output = self.margin - input.mul(target_onehot)
        output = torch.clamp(output, min=0)  # ReLU: max(0, x)
        
        # Apply b-parameter (sharpness)
        if self.b != 1.0:
            output = torch.pow(output + 1e-8, self.b)  # Add epsilon for numerical stability
        
        # Apply β-parameter (temperature/scale)
        loss = self.beta * output.mean()
        
        return loss
    
    def update_b(self, new_b):
        """Update b for b-annealing"""
        self.b = new_b
    
    def update_beta(self, new_beta):
        """Update beta for β-annealing"""
        self.beta = new_beta


# ============================================================================
# MCMC-Inspired Loss Functions
# ============================================================================

class VlogLoss(nn.Module):
    """
    Implements the Vlog potential from MCMC:
    V_log(x, b) = b * (1 - x^(1/b))  if x > 0
                = b * (1 - x)        if x <= 0
    
    For multi-class: stability = (correct_logit - max_wrong_logit) / sqrt(N)
    Total loss = beta * mean(V_log(stabilities, b))
    """
    def __init__(self, b=10.0, beta=1.0, normalization_dim=10):
        super(VlogLoss, self).__init__()
        self.b = b  # tau parameter (kept constant for now)
        self.beta = beta  # inverse temperature (will be annealed)
        self.normalization_dim = normalization_dim
        self.sqrt_n = np.sqrt(normalization_dim)
    
    def vlog_potential(self, x):
        """
        Compute V_log(x, b) element-wise
        x: stability values (can be positive or negative)
        """
        b = self.b
        result = torch.zeros_like(x)
        
        # For x > 0: V = b * (1 - x^(1/b))
        positive_mask = x > 0
        x_pos = x[positive_mask]
        if len(x_pos) > 0:
            # Clamp x^(1/b) to avoid numerical issues
            result[positive_mask] = b * (1.0 - torch.clamp(torch.pow(x_pos, 1.0/b), max=10.0))
        
        # For x <= 0: V = b * (1 - x) 
        negative_mask = ~positive_mask
        result[negative_mask] = b * (1.0 - x[negative_mask])
        
        return result
    
    def compute_stabilities(self, output, target):
        """
        Compute margin-based stabilities for multi-class classification
        stability = (correct_class_score - max_wrong_class_score) / sqrt(N)
        
        Args:
            output: (batch_size, num_classes) - logits or log-probabilities
            target: (batch_size,) - class labels
        Returns:
            stabilities: (batch_size,) - margin values
        """
        batch_size = output.size(0)
        num_classes = output.size(1)
        
        # Get correct class scores
        correct_scores = output[torch.arange(batch_size), target]
        
        # Get max of wrong class scores
        # Create mask to exclude correct class
        mask = torch.ones_like(output).bool()
        mask[torch.arange(batch_size), target] = False
        
        wrong_scores = output.masked_fill(~mask, float('-inf'))
        max_wrong_scores = wrong_scores.max(dim=1)[0]
        
        # Compute margin (stability)
        margins = (correct_scores - max_wrong_scores) / self.sqrt_n
        
        return margins
    
    def forward(self, output, target):
        """
        Compute loss = beta * mean(V_log(stabilities, b))
        """
        stabilities = self.compute_stabilities(output, target)
        potentials = self.vlog_potential(stabilities)
        loss = self.beta * potentials.mean()  # β-annealing enabled!

        return loss
    
    def update_beta(self, new_beta):
        """Update beta for annealing"""
        self.beta = new_beta
    
    def update_b(self, new_b):
        """Update b for b-annealing (tau-annealing)"""
        self.b = new_b


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


class BScheduler:
    """
    Scheduler for b-annealing (tau-annealing): b increases from b_start to b_end
    As b increases, the loss becomes sharper/steeper (like Julia MCMC: 1 → 10^6)
    
    In Julia: tau goes 1 → 10^-6, so b = 1/tau goes 1 → 10^6
    """
    def __init__(self, b_start=1.0, b_end=1000.0, total_epochs=100, schedule_type='exponential'):
        self.b_start = b_start
        self.b_end = b_end
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
    
    def get_b(self, epoch):
        """Get b value for current epoch"""
        if self.schedule_type == 'linear':
            # Linear interpolation
            alpha = epoch / self.total_epochs
            b = self.b_start + (self.b_end - self.b_start) * alpha
        elif self.schedule_type == 'exponential':
            # Exponential growth (default, like Julia MCMC)
            alpha = epoch / self.total_epochs
            b = self.b_start * (self.b_end / self.b_start) ** alpha
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return b


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
    ax2.set_ylim([0, 100])
    
    # Add overall title with experiment details
    if args is not None:
        if args.loss_type == 'ce':
            title = f'Cross-Entropy Loss | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'hinge':
            title = f'Hinge Loss (margin={args.hinge_margin}) | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'hinge_b_annealing':
            title = f'Hinge + b-Annealing (b: {args.hinge_b_start}→{args.hinge_b_end}) | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'hinge_beta_annealing':
            title = f'Hinge + β-Annealing (β: {args.hinge_beta_start}→{args.hinge_beta_end}) | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'hinge_both_annealing':
            title = f'Hinge + BOTH (b: {args.hinge_b_start}→{args.hinge_b_end}, β: {args.hinge_beta_start}→{args.hinge_beta_end}) | Batch={args.batch_size}'
        elif args.loss_type == 'vlog_fixed':
            title = f'Vlog Loss (Fixed β={args.beta_fixed}, b={args.b_value}) | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'vlog_annealing':
            title = f'Vlog Loss (β: {args.beta_start}→{args.beta_end}, b={args.b_value}) | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'vlog_b_annealing':
            title = f'Vlog + b-Annealing (b: {args.vlog_b_start}→{args.vlog_b_end}, β={args.beta_fixed}) | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'vlog_both_annealing':
            title = f'Vlog + BOTH (b: {args.vlog_b_start}→{args.vlog_b_end}, β: {args.beta_start}→{args.beta_end}) | Batch={args.batch_size}'
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

def train(model, device, train_loader, optimizer, criterion, epoch, args, beta_scheduler=None, b_scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    
    # Update beta if using beta-annealing
    if beta_scheduler is not None:
        current_beta = beta_scheduler.get_beta(epoch - 1)  # epoch starts from 1
        criterion.update_beta(current_beta)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: Beta = {current_beta:.4f}')
    
    # Update b if using b-annealing
    if b_scheduler is not None:
        current_b = b_scheduler.get_b(epoch - 1)  # epoch starts from 1
        criterion.update_b(current_b)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: b = {current_b:.4f}')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Different output formats for different losses
        if isinstance(criterion, VlogLoss) or isinstance(criterion, HingeLoss):
            output = model(data, return_logits=True)  # Need raw logits
        else:
            output = model(data, return_logits=False)  # CrossEntropy uses log_softmax
        
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
            
            # Different output formats for different losses
            if isinstance(criterion, VlogLoss) or isinstance(criterion, HingeLoss):
                output = model(data, return_logits=True)  # Need raw logits
            else:
                output = model(data, return_logits=False)  # CrossEntropy uses log_softmax
            
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
    
    # Loss function selection
    parser.add_argument('--loss-type', type=str, default='ce', 
                        choices=['ce', 'hinge', 'hinge_b_annealing', 'hinge_beta_annealing', 'hinge_both_annealing',
                                 'vlog_fixed', 'vlog_annealing', 'vlog_b_annealing', 'vlog_both_annealing'],
                        help='Loss function: ce (CrossEntropy), hinge (standard), hinge_b_annealing (tau-annealing), '
                             'hinge_beta_annealing (temperature-annealing), hinge_both_annealing (both), '
                             'vlog_fixed (Vlog constant), vlog_annealing (Vlog beta-annealing), '
                             'vlog_b_annealing (Vlog b-annealing), vlog_both_annealing (Vlog both)')
    
    # Hinge loss hyperparameters
    parser.add_argument('--hinge-margin', type=float, default=1.0,
                        help='Margin for Hinge loss (default: 1.0)')
    parser.add_argument('--hinge-b-start', type=float, default=1.0,
                        help='Starting b for Hinge b-annealing (default: 1.0)')
    parser.add_argument('--hinge-b-end', type=float, default=100.0,
                        help='Ending b for Hinge b-annealing (default: 100.0)')
    parser.add_argument('--hinge-beta-start', type=float, default=0.5,
                        help='Starting beta for Hinge beta-annealing (default: 0.5)')
    parser.add_argument('--hinge-beta-end', type=float, default=5.0,
                        help='Ending beta for Hinge beta-annealing (default: 5.0)')
    
    # Vlog hyperparameters
    parser.add_argument('--b-value', type=float, default=10.0,
                        help='b (tau) parameter for Vlog potential (default: 10.0)')
    parser.add_argument('--vlog-b-start', type=float, default=1.0,
                        help='Starting b for Vlog b-annealing (default: 1.0)')
    parser.add_argument('--vlog-b-end', type=float, default=100.0,
                        help='Ending b for Vlog b-annealing (default: 100.0)')
    parser.add_argument('--beta-start', type=float, default=0.5,
                        help='Starting beta for annealing (default: 0.5)')
    parser.add_argument('--beta-end', type=float, default=5.0,
                        help='Ending beta for annealing (default: 5.0)')
    parser.add_argument('--beta-fixed', type=float, default=1.0,
                        help='Fixed beta value when not annealing (default: 1.0)')
    parser.add_argument('--normalization-dim', type=int, default=10,
                        help='Dimension for stability normalization (default: 10 for output classes)')
    
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
    
    # Loss function and schedulers
    beta_scheduler = None
    b_scheduler = None
    
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
        print("Using Cross-Entropy Loss")
    
    elif args.loss_type == 'hinge':
        criterion = HingeLoss(margin=args.hinge_margin, b=1.0, beta=1.0).to(device)
        print(f"Using Hinge Loss (margin={args.hinge_margin})")
    
    elif args.loss_type == 'hinge_b_annealing':
        criterion = HingeLoss(margin=args.hinge_margin, b=args.hinge_b_start, beta=1.0).to(device)
        b_scheduler = BScheduler(b_start=args.hinge_b_start,
                                 b_end=args.hinge_b_end,
                                 total_epochs=args.epochs,
                                 schedule_type='exponential')
        print(f"Using Hinge Loss with b-Annealing (b: {args.hinge_b_start} -> {args.hinge_b_end}, margin={args.hinge_margin})")
    
    elif args.loss_type == 'hinge_beta_annealing':
        criterion = HingeLoss(margin=args.hinge_margin, b=1.0, beta=args.hinge_beta_start).to(device)
        beta_scheduler = BetaScheduler(beta_start=args.hinge_beta_start,
                                       beta_end=args.hinge_beta_end,
                                       total_epochs=args.epochs,
                                       schedule_type='linear')
        print(f"Using Hinge Loss with β-Annealing (beta: {args.hinge_beta_start} -> {args.hinge_beta_end}, margin={args.hinge_margin})")
    
    elif args.loss_type == 'hinge_both_annealing':
        criterion = HingeLoss(margin=args.hinge_margin, b=args.hinge_b_start, beta=args.hinge_beta_start).to(device)
        b_scheduler = BScheduler(b_start=args.hinge_b_start,
                                 b_end=args.hinge_b_end,
                                 total_epochs=args.epochs,
                                 schedule_type='exponential')
        beta_scheduler = BetaScheduler(beta_start=args.hinge_beta_start,
                                       beta_end=args.hinge_beta_end,
                                       total_epochs=args.epochs,
                                       schedule_type='linear')
        print(f"Using Hinge Loss with BOTH b-Annealing & β-Annealing")
        print(f"  b: {args.hinge_b_start} -> {args.hinge_b_end} (exponential)")
        print(f"  β: {args.hinge_beta_start} -> {args.hinge_beta_end} (linear)")
    
    elif args.loss_type == 'vlog_fixed':
        criterion = VlogLoss(b=args.b_value, beta=args.beta_fixed, 
                            normalization_dim=args.normalization_dim).to(device)
        print(f"Using Vlog Loss (fixed beta={args.beta_fixed}, b={args.b_value})")
    
    elif args.loss_type == 'vlog_annealing':
        criterion = VlogLoss(b=args.b_value, beta=args.beta_start, 
                            normalization_dim=args.normalization_dim).to(device)
        beta_scheduler = BetaScheduler(beta_start=args.beta_start, 
                                       beta_end=args.beta_end, 
                                       total_epochs=args.epochs,
                                       schedule_type='linear')
        print(f"Using Vlog Loss with Beta-Annealing (beta: {args.beta_start} -> {args.beta_end}, b={args.b_value})")
    
    elif args.loss_type == 'vlog_b_annealing':
        criterion = VlogLoss(b=args.vlog_b_start, beta=args.beta_fixed, 
                            normalization_dim=args.normalization_dim).to(device)
        b_scheduler = BScheduler(b_start=args.vlog_b_start,
                                b_end=args.vlog_b_end,
                                total_epochs=args.epochs,
                                schedule_type='exponential')
        print(f"Using Vlog Loss with b-Annealing (b: {args.vlog_b_start} -> {args.vlog_b_end}, beta={args.beta_fixed})")
    
    elif args.loss_type == 'vlog_both_annealing':
        criterion = VlogLoss(b=args.vlog_b_start, beta=args.beta_start, 
                            normalization_dim=args.normalization_dim).to(device)
        b_scheduler = BScheduler(b_start=args.vlog_b_start,
                                b_end=args.vlog_b_end,
                                total_epochs=args.epochs,
                                schedule_type='exponential')
        beta_scheduler = BetaScheduler(beta_start=args.beta_start,
                                       beta_end=args.beta_end,
                                       total_epochs=args.epochs,
                                       schedule_type='linear')
        print(f"Using Vlog Loss with BOTH b-Annealing & β-Annealing")
        print(f"  b: {args.vlog_b_start} -> {args.vlog_b_end} (exponential)")
        print(f"  β: {args.beta_start} -> {args.beta_end} (linear)")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*70)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Start timing
    import time
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Learning rate decay
        if epoch % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']}")
        
        train_loss, train_acc = train(model, device, train_loader, optimizer, 
                                      criterion, epoch, args, beta_scheduler, b_scheduler)
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
    
    # Create unique experiment name with all key parameters
    experiment_name = f'mnist_{args.loss_type}'
    
    # Add loss-specific parameters
    if args.loss_type == 'hinge':
        experiment_name += f'_m{args.hinge_margin}'
    elif args.loss_type == 'hinge_b_annealing':
        experiment_name += f'_m{args.hinge_margin}_b{args.hinge_b_start}-{args.hinge_b_end}'
    elif args.loss_type == 'hinge_beta_annealing':
        experiment_name += f'_m{args.hinge_margin}_beta{args.hinge_beta_start}-{args.hinge_beta_end}'
    elif args.loss_type == 'hinge_both_annealing':
        experiment_name += f'_m{args.hinge_margin}_b{args.hinge_b_start}-{args.hinge_b_end}_beta{args.hinge_beta_start}-{args.hinge_beta_end}'
    elif args.loss_type.startswith('vlog'):
        if args.loss_type == 'vlog_fixed':
            experiment_name += f'_b{args.b_value}_beta{args.beta_fixed}'
        elif args.loss_type == 'vlog_annealing':
            experiment_name += f'_b{args.b_value}_beta{args.beta_start}-{args.beta_end}'
        elif args.loss_type == 'vlog_b_annealing':
            experiment_name += f'_b{args.vlog_b_start}-{args.vlog_b_end}_beta{args.beta_fixed}'
        elif args.loss_type == 'vlog_both_annealing':
            experiment_name += f'_b{args.vlog_b_start}-{args.vlog_b_end}_beta{args.beta_start}-{args.beta_end}'
    
    # Add training parameters (always included for uniqueness)
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
        f.write(f"Num Workers: {args.num_workers}\n")
        
        if args.loss_type == 'hinge':
            f.write(f"Hinge Margin: {args.hinge_margin}\n")
        elif args.loss_type == 'hinge_b_annealing':
            f.write(f"Hinge Margin: {args.hinge_margin}\n")
            f.write(f"b-annealing: {args.hinge_b_start} -> {args.hinge_b_end} (exponential)\n")
        elif args.loss_type == 'hinge_beta_annealing':
            f.write(f"Hinge Margin: {args.hinge_margin}\n")
            f.write(f"Beta-annealing: {args.hinge_beta_start} -> {args.hinge_beta_end} (linear)\n")
        elif args.loss_type == 'hinge_both_annealing':
            f.write(f"Hinge Margin: {args.hinge_margin}\n")
            f.write(f"b-annealing: {args.hinge_b_start} -> {args.hinge_b_end} (exponential)\n")
            f.write(f"Beta-annealing: {args.hinge_beta_start} -> {args.hinge_beta_end} (linear)\n")
        elif args.loss_type == 'vlog_fixed':
            f.write(f"b value: {args.b_value}\n")
            f.write(f"Fixed beta: {args.beta_fixed}\n")
        elif args.loss_type == 'vlog_annealing':
            f.write(f"b value: {args.b_value}\n")
            f.write(f"Beta annealing: {args.beta_start} -> {args.beta_end} (linear)\n")
        elif args.loss_type == 'vlog_b_annealing':
            f.write(f"b-annealing: {args.vlog_b_start} -> {args.vlog_b_end} (exponential)\n")
            f.write(f"Fixed beta: {args.beta_fixed}\n")
        elif args.loss_type == 'vlog_both_annealing':
            f.write(f"b-annealing: {args.vlog_b_start} -> {args.vlog_b_end} (exponential)\n")
            f.write(f"Beta-annealing: {args.beta_start} -> {args.beta_end} (linear)\n")
        
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

