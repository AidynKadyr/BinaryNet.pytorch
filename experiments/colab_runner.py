"""
Google Colab-friendly experiment runner
Automatically saves plots and results to Google Drive
"""

import subprocess
import os
import sys

def setup_colab():
    """Check if running in Colab and mount Drive if needed"""
    try:
        import google.colab
        in_colab = True
        print("🔍 Detected Google Colab environment")
        
        # Try to mount Google Drive
        try:
            from google.colab import drive
            if not os.path.exists('/content/drive'):
                print("📁 Mounting Google Drive...")
                drive.mount('/content/drive')
            drive_path = '/content/drive/MyDrive/BNN_Experiments'
            os.makedirs(drive_path, exist_ok=True)
            os.makedirs(f'{drive_path}/plots', exist_ok=True)
            os.makedirs(f'{drive_path}/results', exist_ok=True)
            print(f"✅ Google Drive mounted at: {drive_path}")
            return drive_path
        except Exception as e:
            print(f"⚠️ Could not mount Drive: {e}")
            return 'experiments'
    except ImportError:
        print("💻 Running locally (not in Colab)")
        return 'experiments'

def run_experiment(loss_type, epochs=10, batch_size=512, base_dir='experiments', 
                  b_value=10.0, beta_fixed=1.0, beta_start=0.1, beta_end=100.0):
    """Run a single experiment with specified parameters"""
    
    plot_dir = f'{base_dir}/plots'
    
    cmd = [
        'python', 'experiments/mnist_mcmc_experiment.py',
        '--loss-type', loss_type,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--plot-dir', plot_dir,
        '--log-interval', '100'
    ]
    
    if loss_type == 'vlog_fixed':
        cmd.extend(['--b-value', str(b_value)])
        cmd.extend(['--beta-fixed', str(beta_fixed)])
    elif loss_type == 'vlog_annealing':
        cmd.extend(['--b-value', str(b_value)])
        cmd.extend(['--beta-start', str(beta_start)])
        cmd.extend(['--beta-end', str(beta_end)])
    
    print(f"\n{'='*80}")
    print(f"🚀 Running: {loss_type}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✅ Experiment '{loss_type}' completed successfully!")
        return True
    else:
        print(f"❌ Experiment '{loss_type}' failed!")
        return False

def main():
    print("\n" + "="*80)
    print("🔬 MNIST Binary Network Experiments")
    print("="*80 + "\n")
    
    # Setup paths
    base_dir = setup_colab()
    
    # Configuration
    print("\n📋 Configuration:")
    epochs = 10  # Start with quick test
    batch_size = 512
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Save location: {base_dir}")
    print()
    
    # Ask user which experiments to run
    print("Which experiments would you like to run?")
    print("  1. Cross-Entropy (baseline)")
    print("  2. Vlog (fixed b=10)")
    print("  3. Both (recommended)")
    print("  4. All three (including vlog_annealing - currently same as #2)")
    
    choice = input("\nEnter choice (1-4) [default: 3]: ").strip() or "3"
    
    experiments = []
    if choice == "1":
        experiments = ['ce']
    elif choice == "2":
        experiments = ['vlog_fixed']
    elif choice == "3":
        experiments = ['ce', 'vlog_fixed']
    elif choice == "4":
        experiments = ['ce', 'vlog_fixed', 'vlog_annealing']
    else:
        print(f"Invalid choice '{choice}', running CE and Vlog")
        experiments = ['ce', 'vlog_fixed']
    
    # Run experiments
    results = {}
    for exp in experiments:
        success = run_experiment(exp, epochs=epochs, batch_size=batch_size, base_dir=base_dir)
        results[exp] = success
    
    # Summary
    print("\n" + "="*80)
    print("📊 EXPERIMENT SUMMARY")
    print("="*80)
    for exp, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {exp}: {status}")
    
    print(f"\n📁 Results and plots saved to: {base_dir}")
    print(f"   Plots: {base_dir}/plots/")
    print(f"   Results: {base_dir}/results/")
    
    if base_dir.startswith('/content/drive'):
        print(f"\n💾 Files are saved in your Google Drive!")
        print(f"   You can access them at: Google Drive/BNN_Experiments/")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

