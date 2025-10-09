"""
Compare results from different loss function experiments
"""

import os
import re

def parse_results(filename):
    """Parse a results file and extract key metrics"""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract metrics using regex
    loss_type = re.search(r'Loss Type: (\w+)', content)
    final_acc = re.search(r'Final Test Accuracy: ([\d.]+)%', content)
    best_acc = re.search(r'Best Test Accuracy: ([\d.]+)% \(Epoch (\d+)\)', content)
    
    if not (loss_type and final_acc and best_acc):
        return None
    
    return {
        'loss_type': loss_type.group(1),
        'final_accuracy': float(final_acc.group(1)),
        'best_accuracy': float(best_acc.group(1)),
        'best_epoch': int(best_acc.group(2))
    }

def main():
    results_dir = 'experiments/results'
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Run experiments first!")
        return
    
    # Parse all results files
    experiments = ['ce', 'vlog_fixed', 'vlog_annealing']
    results = {}
    
    for exp in experiments:
        filename = os.path.join(results_dir, f'mnist_{exp}_results.txt')
        parsed = parse_results(filename)
        if parsed:
            results[exp] = parsed
        else:
            print(f"Warning: Could not parse {filename}")
    
    if not results:
        print("No results found. Run experiments first!")
        return
    
    # Display comparison
    print("\n" + "="*80)
    print("MNIST BINARY NETWORK - RESULTS COMPARISON")
    print("="*80)
    print()
    
    # Table header
    print(f"{'Loss Function':<25} {'Final Acc':<12} {'Best Acc':<12} {'Best Epoch':<12}")
    print("-" * 80)
    
    # Table rows
    names = {
        'ce': 'Cross-Entropy',
        'vlog_fixed': 'Vlog (fixed β)',
        'vlog_annealing': 'Vlog (β-annealing)'
    }
    
    for exp in experiments:
        if exp in results:
            r = results[exp]
            print(f"{names[exp]:<25} {r['final_accuracy']:<12.2f} {r['best_accuracy']:<12.2f} {r['best_epoch']:<12}")
    
    print()
    
    # Determine winner
    if len(results) > 1:
        best_exp = max(results.items(), key=lambda x: x[1]['best_accuracy'])
        print(f"🏆 Best Performance: {names[best_exp[0]]} with {best_exp[1]['best_accuracy']:.2f}% accuracy")
        
        # Compare to baseline
        if 'ce' in results and best_exp[0] != 'ce':
            improvement = best_exp[1]['best_accuracy'] - results['ce']['best_accuracy']
            print(f"   Improvement over Cross-Entropy: {improvement:+.2f}%")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()

