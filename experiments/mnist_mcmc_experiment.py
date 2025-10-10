"""
MNIST Binary Network with MCMC-inspired Loss Functions
Compares:
1. Standard Cross-Entropy
2. Vlog potential (fixed beta, constant b)
3. Vlog potential with beta-annealing (beta: small -> large)
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
        #loss = self.beta * potentials.mean()
        loss = potentials.mean()

        return loss
    
    def update_beta(self, new_beta):
        """Update beta for annealing"""
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
    ax2.set_ylim([0, 100])
    
    # Add overall title with experiment details
    if args is not None:
        if args.loss_type == 'ce':
            title = f'Cross-Entropy Loss | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'vlog_fixed':
            title = f'Vlog Loss (Fixed β={args.beta_fixed}, b={args.b_value}) | LR={args.lr} | Batch={args.batch_size}'
        elif args.loss_type == 'vlog_annealing':
            title = f'Vlog Loss (β: {args.beta_start}→{args.beta_end}, b={args.b_value}) | LR={args.lr} | Batch={args.batch_size}'
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
    
    # Update beta if using annealing
    if beta_scheduler is not None and isinstance(criterion, VlogLoss):
        current_beta = beta_scheduler.get_beta(epoch - 1)  # epoch starts from 1
        criterion.update_beta(current_beta)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: Beta = {current_beta:.4f}')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # For Vlog loss, we need raw logits; for CrossEntropy, we need log_softmax
        if isinstance(criterion, VlogLoss):
            output = model(data, return_logits=True)
        else:
            output = model(data, return_logits=False)
        
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
        
        # Calculate accuracy
        if isinstance(criterion, VlogLoss):
            pred = output.argmax(dim=1)
        else:
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
            
            # For Vlog loss, we need raw logits; for CrossEntropy, we need log_softmax
            if isinstance(criterion, VlogLoss):
                output = model(data, return_logits=True)
            else:
                output = model(data, return_logits=False)
            
            test_loss += criterion(output, target).item()
            
            # Get predictions
            if isinstance(criterion, VlogLoss):
                pred = output.argmax(dim=1)
            else:
                pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader.dataset)  # Match main_mnist.py: divide by number of samples
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
    
    # Loss function selection
    parser.add_argument('--loss-type', type=str, default='ce', 
                        choices=['ce', 'vlog_fixed', 'vlog_annealing'],
                        help='Loss function: ce (CrossEntropy), vlog_fixed (Vlog with constant beta), vlog_annealing (Vlog with beta annealing)')
    
    # Vlog hyperparameters
    parser.add_argument('--b-value', type=float, default=10.0,
                        help='b (tau) parameter for Vlog potential (default: 10.0)')
    parser.add_argument('--beta-start', type=float, default=0.1,
                        help='Starting beta for annealing (default: 0.1)')
    parser.add_argument('--beta-end', type=float, default=100.0,
                        help='Ending beta for annealing (default: 100.0)')
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
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
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
    
    # Loss function and scheduler
    beta_scheduler = None
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
        print("Using Cross-Entropy Loss")
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
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*70)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(1, args.epochs + 1):
        # Learning rate decay
        if epoch % 40 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']}")
        
        train_loss, train_acc = train(model, device, train_loader, optimizer, 
                                      criterion, epoch, args, beta_scheduler)
        test_loss, test_acc = test(model, device, test_loader, criterion, args)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    # Final summary
    print("="*70)
    print("Training Complete!")
    print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"Best Test Accuracy: {max(test_accs):.2f}% (Epoch {test_accs.index(max(test_accs))+1})")
    
    # Plot training curves
    if not args.no_plot:
        experiment_name = f'mnist_{args.loss_type}'
        if args.loss_type.startswith('vlog'):
            experiment_name += f'_b{args.b_value}'
            if args.loss_type == 'vlog_annealing':
                experiment_name += f'_beta{args.beta_start}-{args.beta_end}'
            else:
                experiment_name += f'_beta{args.beta_fixed}'
        plot_training_curves(train_losses, test_losses, train_accs, test_accs,
                           experiment_name, save_dir=args.plot_dir, args=args)
    
    # Save results
    results_dir = 'experiments/results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'mnist_{args.loss_type}_results.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Loss Type: {args.loss_type}\n")
        f.write(f"Epochs: {args.epochs}\n")
        if args.loss_type.startswith('vlog'):
            f.write(f"b value: {args.b_value}\n")
            if args.loss_type == 'vlog_annealing':
                f.write(f"Beta annealing: {args.beta_start} -> {args.beta_end}\n")
            else:
                f.write(f"Fixed beta: {args.beta_fixed}\n")
        f.write(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%\n")
        f.write(f"Best Test Accuracy: {max(test_accs):.2f}% (Epoch {test_accs.index(max(test_accs))+1})\n")
        f.write(f"\nTest Accuracies per epoch:\n")
        for i, acc in enumerate(test_accs):
            f.write(f"Epoch {i+1}: {acc:.2f}%\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()

