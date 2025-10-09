"""
Run all three experiments for comparison:
1. Cross-Entropy (baseline)
2. Vlog with fixed beta
3. Vlog with beta-annealing
"""

import subprocess
import sys
import os

def run_experiment(loss_type, additional_args=""):
    """Run a single experiment"""
    print(f"\n{'='*80}")
    print(f"Running experiment: {loss_type}")
    print(f"{'='*80}\n")
    
    cmd = f"python experiments/mnist_mcmc_experiment.py --loss-type {loss_type} {additional_args}"
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\nERROR: Experiment {loss_type} failed!")
        return False
    return True

def main():
    # Default parameters
    epochs = 100
    b_value = 10.0
    beta_fixed = 1.0
    beta_start = 0.1
    beta_end = 100.0
    
    # Parse command line for quick parameter changes
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    
    base_args = f"--epochs {epochs} --log-interval 100"
    
    print("\n" + "="*80)
    print("MNIST Binary Network - MCMC Loss Comparison")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Epochs: {epochs}")
    print(f"  b (tau): {b_value}")
    print(f"  Beta (fixed): {beta_fixed}")
    print(f"  Beta (annealing): {beta_start} -> {beta_end}")
    print("\n" + "="*80)
    
    # Experiment 1: Cross-Entropy baseline
    success = run_experiment("ce", base_args)
    if not success:
        print("\nStopping due to error.")
        return
    
    # Experiment 2: Vlog with fixed beta
    vlog_fixed_args = f"{base_args} --b-value {b_value} --beta-fixed {beta_fixed}"
    success = run_experiment("vlog_fixed", vlog_fixed_args)
    if not success:
        print("\nStopping due to error.")
        return
    
    # Experiment 3: Vlog with beta-annealing
    vlog_annealing_args = f"{base_args} --b-value {b_value} --beta-start {beta_start} --beta-end {beta_end}"
    success = run_experiment("vlog_annealing", vlog_annealing_args)
    if not success:
        print("\nStopping due to error.")
        return
    
    # Print summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nResults saved in experiments/results/")
    print("\nTo compare results:")
    print("  cat experiments/results/mnist_*_results.txt")
    print("\n")

if __name__ == "__main__":
    main()

