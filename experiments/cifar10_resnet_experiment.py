"""
CIFAR-10 Binary ResNet with MCMC-inspired Loss Functions
Same experimental pipeline as CIFAR-10 VGG but using ResNet architecture

Compares:
1. Standard Cross-Entropy
2. HingeLoss (SVM-style margin-based) with all annealing variants
3. Vlog potential (fixed, β-annealing, b-annealing, both)
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
from models.resnet_binary import ResNet_cifar10
from models.binarized_modules import Binarize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures


# ============================================================================
# Import Loss Functions and Schedulers from MNIST experiment
# ============================================================================

from mnist_mcmc_experiment import (
    HingeLoss, 
    VlogLoss, 
    BetaScheduler, 
    BScheduler,
    plot_training_curves
)


# ============================================================================
# Modified ResNet Model with return_logits support
# ============================================================================

class ResNet_cifar10_Logits(ResNet_cifar10):
    """
    Extended ResNet_cifar10 with return_logits option for Vlog/Hinge losses.
    Overrides forward() to optionally skip the final LogSoftmax.
    """
    def forward(self, x, return_logits=False):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)

        if return_logits:
            return x  # Return raw logits for Vlog/Hinge loss
        return self.logsoftmax(x)


# ============================================================================
# Training and Testing
# ============================================================================

def train(model, device, train_loader, optimizer, criterion, epoch, args, beta_scheduler=None, b_scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    
    # Update beta if using beta-annealing
    if beta_scheduler is not None:
        current_beta = beta_scheduler.get_beta(epoch - 1)
        criterion.update_beta(current_beta)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: Beta = {current_beta:.4f}')
    
    # Update b if using b-annealing
    if b_scheduler is not None:
        current_b = b_scheduler.get_b(epoch - 1)
        criterion.update_b(current_b)
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: b = {current_b:.4f}')
    
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
        
        # Calculate accuracy
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
            
            if isinstance(criterion, HingeLoss):
                num_classes = output.size(1)
                target_onehot = torch.zeros(target.size(0), num_classes, device=device)
                target_onehot.fill_(-1)
                target_onehot.scatter_(1, target.unsqueeze(1), 1)
                test_loss += criterion(output, target_onehot).item()
            else:
                test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Binary ResNet with MCMC Loss')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--lr-decay-factor', type=float, default=0.2,
                        help='multiply LR by this factor at each decay step (default: 0.2)')
    parser.add_argument('--lr-decay-steps', type=int, default=4,
                        help='number of LR decay steps during training (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of data loading workers (default: 4)')
    
    # Loss function selection
    parser.add_argument('--loss-type', type=str, default='ce', 
                        choices=['ce', 'hinge', 'hinge_b_annealing', 'hinge_beta_annealing', 'hinge_both_annealing',
                                 'vlog_fixed', 'vlog_annealing', 'vlog_b_annealing', 'vlog_both_annealing'],
                        help='Loss function type')
    
    # Hinge loss hyperparameters
    parser.add_argument('--hinge-margin', type=float, default=1.0)
    parser.add_argument('--hinge-b-start', type=float, default=1.0)
    parser.add_argument('--hinge-b-end', type=float, default=100.0)
    parser.add_argument('--hinge-beta-start', type=float, default=0.5)
    parser.add_argument('--hinge-beta-end', type=float, default=5.0)
    
    # Vlog hyperparameters
    parser.add_argument('--b-value', type=float, default=10.0)
    parser.add_argument('--vlog-b-start', type=float, default=1.0)
    parser.add_argument('--vlog-b-end', type=float, default=100.0)
    parser.add_argument('--beta-start', type=float, default=0.5)
    parser.add_argument('--beta-end', type=float, default=5.0)
    parser.add_argument('--beta-fixed', type=float, default=1.0)
    parser.add_argument('--normalization-dim', type=int, default=10)
    
    # Plotting options
    parser.add_argument('--plot-dir', type=str, default='experiments/plots')
    parser.add_argument('--no-plot', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    if use_cuda:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (training will be VERY slow!)")
    
    # CIFAR-10 data loaders
    print("Loading CIFAR-10 dataset...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform_test),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    # Model
    print("Creating ResNet-18 BinaryNet model (inflate=5)...")
    model = ResNet_cifar10_Logits(num_classes=10).to(device)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss function and schedulers setup
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
        b_scheduler = BScheduler(b_start=args.hinge_b_start, b_end=args.hinge_b_end,
                                total_epochs=args.epochs, schedule_type='exponential')
        print(f"Using Hinge + b-Annealing (b: {args.hinge_b_start}->{args.hinge_b_end})")
    
    elif args.loss_type == 'hinge_beta_annealing':
        criterion = HingeLoss(margin=args.hinge_margin, b=1.0, beta=args.hinge_beta_start).to(device)
        beta_scheduler = BetaScheduler(beta_start=args.hinge_beta_start, beta_end=args.hinge_beta_end,
                                       total_epochs=args.epochs, schedule_type='linear')
        print(f"Using Hinge + beta-Annealing (beta: {args.hinge_beta_start}->{args.hinge_beta_end})")
    
    elif args.loss_type == 'hinge_both_annealing':
        criterion = HingeLoss(margin=args.hinge_margin, b=args.hinge_b_start, beta=args.hinge_beta_start).to(device)
        b_scheduler = BScheduler(b_start=args.hinge_b_start, b_end=args.hinge_b_end,
                                total_epochs=args.epochs, schedule_type='exponential')
        beta_scheduler = BetaScheduler(beta_start=args.hinge_beta_start, beta_end=args.hinge_beta_end,
                                       total_epochs=args.epochs, schedule_type='linear')
        print(f"Using Hinge + BOTH Annealing")
    
    elif args.loss_type == 'vlog_fixed':
        criterion = VlogLoss(b=args.b_value, beta=args.beta_fixed, 
                            normalization_dim=args.normalization_dim).to(device)
        print(f"Using Vlog Loss (fixed b={args.b_value}, beta={args.beta_fixed})")
    
    elif args.loss_type == 'vlog_annealing':
        criterion = VlogLoss(b=args.b_value, beta=args.beta_start, 
                            normalization_dim=args.normalization_dim).to(device)
        beta_scheduler = BetaScheduler(beta_start=args.beta_start, beta_end=args.beta_end,
                                       total_epochs=args.epochs, schedule_type='linear')
        print(f"Using Vlog + beta-Annealing (beta: {args.beta_start}->{args.beta_end}, b={args.b_value})")
    
    elif args.loss_type == 'vlog_b_annealing':
        criterion = VlogLoss(b=args.vlog_b_start, beta=args.beta_fixed,
                            normalization_dim=args.normalization_dim).to(device)
        b_scheduler = BScheduler(b_start=args.vlog_b_start, b_end=args.vlog_b_end,
                                total_epochs=args.epochs, schedule_type='exponential')
        print(f"Using Vlog + b-Annealing (b: {args.vlog_b_start}->{args.vlog_b_end})")
    
    elif args.loss_type == 'vlog_both_annealing':
        criterion = VlogLoss(b=args.vlog_b_start, beta=args.beta_start,
                            normalization_dim=args.normalization_dim).to(device)
        b_scheduler = BScheduler(b_start=args.vlog_b_start, b_end=args.vlog_b_end,
                                total_epochs=args.epochs, schedule_type='exponential')
        beta_scheduler = BetaScheduler(beta_start=args.beta_start, beta_end=args.beta_end,
                                       total_epochs=args.epochs, schedule_type='linear')
        print(f"Using Vlog + BOTH Annealing")
    
    # Optimizer (Adam, matching ResNet regime)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    # Compute LR decay milestones (evenly spaced)
    lr_milestones = set()
    for step in range(1, args.lr_decay_steps + 1):
        milestone = int(args.epochs * step / (args.lr_decay_steps + 1))
        lr_milestones.add(milestone)
    
    # Print schedule preview
    lr_preview = args.lr
    schedule_parts = [f"{args.lr}"]
    for m in sorted(lr_milestones):
        lr_preview *= args.lr_decay_factor
        schedule_parts.append(f"{lr_preview:.1e} (ep{m})")
    print(f"LR schedule: {' -> '.join(schedule_parts)}")
    print("="*70)
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    import time
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Learning rate schedule (automatic decay)
        if epoch in lr_milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_factor
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate -> {current_lr:.1e}")
        
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
    
    # Final summary
    print("="*70)
    print("Training Complete!")
    print(f"Training Time: {training_time:.1f}s ({training_time/60:.2f} min)")
    print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"Best Test Accuracy: {max(test_accs):.2f}% (Epoch {test_accs.index(max(test_accs))+1})")
    
    # Create experiment name
    experiment_name = f'cifar10_resnet_{args.loss_type}'
    
    # Add hyperparameters to name
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
    
    experiment_name += f'_e{args.epochs}_bs{args.batch_size}_lr{args.lr}'
    
    # Plot training curves
    if not args.no_plot:
        plot_training_curves(train_losses, test_losses, train_accs, test_accs,
                           experiment_name, save_dir=args.plot_dir, args=args)
    
    # Save results
    if args.plot_dir.endswith('plots'):
        base_dir = os.path.dirname(args.plot_dir)
    else:
        base_dir = args.plot_dir
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'{experiment_name}.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"="*60 + "\n\n")
        
        f.write(f"CONFIGURATION:\n")
        f.write(f"Dataset: CIFAR-10\n")
        f.write(f"Model: ResNet-18 BinaryNet (inflate=5, depth=18)\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Loss Type: {args.loss_type}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"LR Decay Factor: {args.lr_decay_factor} at {args.lr_decay_steps} steps (milestones: {sorted(lr_milestones)})\n")
        f.write(f"Num Workers: {args.num_workers}\n")
        
        # Loss-specific params
        if args.loss_type.startswith('hinge'):
            f.write(f"Hinge Margin: {args.hinge_margin}\n")
            if 'b_annealing' in args.loss_type:
                f.write(f"b-annealing: {args.hinge_b_start} -> {args.hinge_b_end}\n")
            if 'beta_annealing' in args.loss_type:
                f.write(f"beta-annealing: {args.hinge_beta_start} -> {args.hinge_beta_end}\n")
        elif args.loss_type.startswith('vlog'):
            if args.loss_type == 'vlog_fixed':
                f.write(f"b value: {args.b_value}\n")
                f.write(f"Fixed beta: {args.beta_fixed}\n")
            elif args.loss_type == 'vlog_annealing':
                f.write(f"b value: {args.b_value}\n")
                f.write(f"beta-annealing: {args.beta_start} -> {args.beta_end}\n")
            elif args.loss_type == 'vlog_b_annealing':
                f.write(f"b-annealing: {args.vlog_b_start} -> {args.vlog_b_end}\n")
                f.write(f"Fixed beta: {args.beta_fixed}\n")
            elif args.loss_type == 'vlog_both_annealing':
                f.write(f"b-annealing: {args.vlog_b_start} -> {args.vlog_b_end}\n")
                f.write(f"beta-annealing: {args.beta_start} -> {args.beta_end}\n")
        
        f.write(f"\nTRAINING TIME:\n")
        f.write(f"Total Time: {training_time:.1f}s ({training_time/60:.2f} min)\n")
        f.write(f"Time per Epoch: {training_time/args.epochs:.1f}s\n")
        
        f.write(f"\nRESULTS:\n")
        f.write(f"Final Test Accuracy: {test_accs[-1]:.2f}%\n")
        f.write(f"Best Test Accuracy: {max(test_accs):.2f}% (Epoch {test_accs.index(max(test_accs))+1})\n")
        
        f.write(f"\nPER-EPOCH RESULTS:\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Test Loss':<15} {'Train Acc':<12} {'Test Acc':<12}\n")
        f.write(f"{'-'*70}\n")
        for i in range(len(test_accs)):
            f.write(f"{i+1:<8} {train_losses[i]:<15.6f} {test_losses[i]:<15.6f} {train_accs[i]:<12.2f}% {test_accs[i]:<12.2f}%\n")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
