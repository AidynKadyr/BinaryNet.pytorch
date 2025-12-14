"""
Comprehensive Analysis Script for Binary Neural Network Experiments

This script:
1. Parses all result files in experiments/results/
2. Extracts key metrics (accuracy, training time, hyperparameters)
3. Generates comparison tables ORGANIZED BY DATASET (MNIST vs CIFAR-10)
4. Creates an HTML report with embedded plots
5. Provides ranking and insights per dataset
6. Enables cross-dataset comparison for similar experiment configurations

Usage (from summary_results folder):
    python analyze_all_results.py
    
    # Or from experiments folder:
    python summary_results/analyze_all_results.py
    
    # Custom output format:
    python analyze_all_results.py --output-format html
    python analyze_all_results.py --output-format markdown
"""

import os
import re
import glob
import argparse
from pathlib import Path
from collections import defaultdict
import base64


class ExperimentResult:
    """Container for a single experiment's results"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        # Detect dataset from filename
        if self.filename.startswith('cifar10_'):
            self.dataset = 'cifar10'
        elif self.filename.startswith('mnist_'):
            self.dataset = 'mnist'
        else:
            self.dataset = 'unknown'
        self.parse_results()
    
    def parse_results(self):
        """Parse result file and extract all metrics"""
        with open(self.filepath, 'r') as f:
            content = f.read()
        
        # Extract experiment name
        match = re.search(r'Experiment: (.+)', content)
        self.experiment_name = match.group(1) if match else "Unknown"
        
        # Extract configuration
        self.loss_type = self._extract(content, r'Loss Type: (\S+)')
        self.epochs = self._extract_int(content, r'Epochs: (\d+)')
        self.batch_size = self._extract_int(content, r'Batch Size: (\d+)')
        self.lr = self._extract_float(content, r'Learning Rate: ([\d.]+)')
        self.num_workers = self._extract_int(content, r'Num Workers: (\d+)')
        
        # Loss-specific parameters
        self.hinge_margin = self._extract_float(content, r'Hinge Margin: ([\d.]+)')
        self.b_value = self._extract_float(content, r'b value: ([\d.]+)')
        self.beta_fixed = self._extract_float(content, r'Fixed beta: ([\d.]+)')
        self.b_annealing = self._extract(content, r'b-annealing: ([\d.]+ -> [\d.]+)')
        self.beta_annealing = self._extract(content, r'[Bb]eta.?annealing: ([\d.]+ -> [\d.]+)')
        
        # Training time
        self.total_time_sec = self._extract_float(content, r'Total Time: ([\d.]+)s')
        self.time_per_epoch = self._extract_float(content, r'Time per Epoch: ([\d.]+)s')
        
        # Results
        self.final_accuracy = self._extract_float(content, r'Final Test Accuracy: ([\d.]+)%')
        best_match = re.search(r'Best Test Accuracy: ([\d.]+)% \(Epoch (\d+)\)', content)
        if best_match:
            self.best_accuracy = float(best_match.group(1))
            self.best_epoch = int(best_match.group(2))
        else:
            self.best_accuracy = None
            self.best_epoch = None
        
        # Check if experiment failed (accuracy < 50% usually means failure)
        self.failed = self.final_accuracy is not None and self.final_accuracy < 50.0
        
        # Find corresponding plot file
        self.plot_file = self._find_plot_file()
    
    def _extract(self, content, pattern):
        """Extract string using regex"""
        match = re.search(pattern, content)
        return match.group(1) if match else None
    
    def _extract_int(self, content, pattern):
        """Extract integer using regex"""
        result = self._extract(content, pattern)
        return int(result) if result else None
    
    def _extract_float(self, content, pattern):
        """Extract float using regex"""
        result = self._extract(content, pattern)
        return float(result) if result else None
    
    def _find_plot_file(self):
        """Find corresponding plot file"""
        # Remove .txt extension and add .png
        base_name = self.experiment_name
        
        # Try different possible directories
        possible_dirs = [
            '../plots',
            'plots',
            '../experiments/plots',
            'experiments/plots',
        ]
        
        for plot_dir in possible_dirs:
            plot_path = os.path.join(plot_dir, f'{base_name}.png')
            if os.path.exists(plot_path):
                # Convert to forward slashes for cross-platform compatibility (especially GitHub)
                return plot_path.replace('\\', '/')
        
        return None
    
    def get_loss_description(self):
        """Get human-readable loss description"""
        descriptions = {
            'ce': 'Cross-Entropy',
            'hinge': 'Hinge Loss',
            'hinge_b_annealing': 'Hinge + b-Annealing',
            'hinge_beta_annealing': 'Hinge + β-Annealing',
            'hinge_both_annealing': 'Hinge + Both Annealing',
            'vlog_fixed': 'Vlog (Fixed)',
            'vlog_annealing': 'Vlog + β-Annealing',
            'vlog_b_annealing': 'Vlog + b-Annealing',
            'vlog_both_annealing': 'Vlog + Both Annealing',
        }
        return descriptions.get(self.loss_type, self.loss_type)
    
    def get_hyperparameters_str(self):
        """Get hyperparameters as string"""
        params = []
        
        if self.hinge_margin:
            params.append(f"margin={self.hinge_margin}")
        
        if self.b_value:
            params.append(f"b={self.b_value}")
        
        if self.beta_fixed:
            params.append(f"β={self.beta_fixed}")
        
        if self.b_annealing:
            params.append(f"b: {self.b_annealing}")
        
        if self.beta_annealing:
            params.append(f"β: {self.beta_annealing}")
        
        return ", ".join(params) if params else "—"


def collect_all_results(results_dir):
    """Collect and parse all result files"""
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return []
    
    result_files = glob.glob(os.path.join(results_dir, '*.txt'))
    
    if not result_files:
        print(f"Warning: No result files found in {results_dir}")
        return []
    
    results = []
    for filepath in sorted(result_files):
        try:
            result = ExperimentResult(filepath)
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not parse {filepath}: {e}")
    
    return results


def generate_summary_table(results):
    """Generate summary comparison table"""
    # Group by loss type and epochs
    groups = defaultdict(list)
    for r in results:
        key = (r.loss_type, r.epochs)
        groups[key].append(r)
    
    # Sort groups by loss type, then epochs
    sorted_groups = sorted(groups.items(), key=lambda x: (x[0][0], x[0][1]))
    
    return sorted_groups, groups


def group_by_dataset(results):
    """Group results by dataset"""
    datasets = defaultdict(list)
    for r in results:
        datasets[r.dataset].append(r)
    return datasets


def get_experiment_config_key(result):
    """Get a normalized configuration key for cross-dataset comparison"""
    # Create a key based on loss type and key hyperparameters (ignoring dataset-specific epochs/lr)
    key_parts = [result.loss_type]
    
    if result.hinge_margin:
        key_parts.append(f"m{result.hinge_margin}")
    if result.b_value:
        key_parts.append(f"b{result.b_value}")
    if result.beta_fixed:
        key_parts.append(f"beta{result.beta_fixed}")
    if result.b_annealing:
        key_parts.append(f"b_ann:{result.b_annealing}")
    if result.beta_annealing:
        key_parts.append(f"beta_ann:{result.beta_annealing}")
    
    return "_".join(key_parts)


def find_cross_dataset_comparisons(results):
    """Find experiments with similar configurations across datasets"""
    # Group by config key
    config_groups = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        if r.failed:
            continue
        config_key = get_experiment_config_key(r)
        config_groups[config_key][r.dataset].append(r)
    
    # Filter to only configs that appear in multiple datasets
    cross_dataset = {}
    for config_key, dataset_results in config_groups.items():
        if len(dataset_results) > 1:  # Present in multiple datasets
            cross_dataset[config_key] = dataset_results
    
    return cross_dataset


def generate_markdown_report(results, output_file):
    """Generate markdown report with simple tables of all experiments grouped by dataset"""
    
    # Group by dataset
    datasets = group_by_dataset(results)
    dataset_names = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'unknown': 'Unknown'}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Binary Neural Network - Experimental Results\n\n")
        f.write(f"**Total experiments: {len(results)}**\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        for dataset in sorted(datasets.keys()):
            ds_name = dataset_names.get(dataset, dataset)
            f.write(f"- [{ds_name}](#{dataset.lower()})\n")
        f.write("- [Training Curves](#training-curves)\n")
        f.write("\n---\n\n")
        
        # ============== DATASET TABLES ==============
        for dataset in sorted(datasets.keys()):
            ds_name = dataset_names.get(dataset, dataset)
            dataset_results = datasets[dataset]
            
            # Sort by filename
            dataset_results = sorted(dataset_results, key=lambda x: x.filename)
            
            successful = len([r for r in dataset_results if not r.failed])
            failed = len([r for r in dataset_results if r.failed])
            
            f.write(f"# {ds_name}\n\n")
            f.write(f"**{len(dataset_results)} experiments** ({successful} successful, {failed} failed)\n\n")
            
            # Single table with all experiments (filename first)
            f.write("| Filename | Loss | Best Acc | Final Acc | Epochs | BS | LR | Time | Status |\n")
            f.write("|----------|------|----------|-----------|--------|----|----|------|--------|\n")
            
            for r in dataset_results:
                status = "❌" if r.failed else "✅"
                best_acc = f"**{r.best_accuracy:.2f}%**" if r.best_accuracy else "N/A"
                final_acc = f"{r.final_accuracy:.2f}%" if r.final_accuracy else "N/A"
                time_str = f"{r.total_time_sec/60:.1f}min" if r.total_time_sec else "N/A"
                lr_str = f"{r.lr}" if r.lr else "N/A"
                bs_str = f"{r.batch_size}" if r.batch_size else "N/A"
                # Remove .txt extension for cleaner display
                filename = r.filename.replace('.txt', '')
                # Get base loss type (ce, hinge, vlog)
                loss_type = r.loss_type.split('_')[0] if r.loss_type else "N/A"
                
                f.write(f"| `{filename}` | {loss_type} | {best_acc} | {final_acc} | {r.epochs} | {bs_str} | {lr_str} | {time_str} | {status} |\n")
            
            f.write("\n---\n\n")
        
        # ============== TRAINING CURVES ==============
        f.write("# Training Curves\n\n")
        
        for dataset in sorted(datasets.keys()):
            ds_name = dataset_names.get(dataset, dataset)
            dataset_results = datasets[dataset]
            # Sort by filename
            dataset_results = sorted(dataset_results, key=lambda x: x.filename)
            
            f.write(f"## {ds_name}\n\n")
            
            for r in dataset_results:
                if r.plot_file and os.path.exists(r.plot_file):
                    status = "❌" if r.failed else "✅"
                    acc_str = f"{r.best_accuracy:.2f}%" if r.best_accuracy else f"{r.final_accuracy:.2f}%"
                    filename = r.filename.replace('.txt', '')
                    f.write(f"### `{filename}` | {acc_str} {status}\n\n")
                    f.write(f"![{r.experiment_name}]({r.plot_file})\n\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("*Generated by `analyze_all_results.py`*\n")
    
    print(f"✅ Markdown report saved to: {output_file}")
    return output_file


def generate_html_report(results, output_file):
    """Generate HTML report with simple tables of all experiments grouped by dataset"""
    
    # Group by dataset
    datasets = group_by_dataset(results)
    dataset_names = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'unknown': 'Unknown'}
    dataset_colors = {'mnist': '#9b59b6', 'cifar10': '#e67e22', 'unknown': '#95a5a6'}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BNN Experimental Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .nav-tabs {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .nav-tab {
            padding: 10px 20px;
            background: #ecf0f1;
            border-radius: 5px;
            text-decoration: none;
            color: #2c3e50;
        }
        .nav-tab:hover { background: #3498db; color: white; }
        .dataset-section {
            margin: 30px 0;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid;
        }
        .dataset-mnist { background: #f5eef8; border-color: #9b59b6; }
        .dataset-cifar10 { background: #fef5e7; border-color: #e67e22; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            font-size: 0.9em;
        }
        th {
            background: #3498db;
            color: white;
            padding: 10px 8px;
            text-align: left;
            font-weight: 600;
        }
        th.mnist-header { background: #9b59b6; }
        th.cifar-header { background: #e67e22; }
        td {
            padding: 8px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover { background: #f8f9fa; }
        .success { color: #27ae60; }
        .failed { color: #e74c3c; }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .comparison-box {
            background: #e3f2fd;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Binary Neural Network - Experimental Results</h1>
        <p><strong>Total Experiments:</strong> """ + str(len(results)) + """</p>
        
        <div class="nav-tabs">
""")
        
        for dataset in sorted(datasets.keys()):
            ds_name = dataset_names.get(dataset, dataset)
            f.write(f"<a class='nav-tab' href='#{dataset}-section'>{ds_name}</a>\n")
        f.write("<a class='nav-tab' href='#curves-section'>Curves</a>\n")
        f.write("</div>\n<hr>\n")
        
        # ============== DATASET TABLES ==============
        for dataset in sorted(datasets.keys()):
            ds_name = dataset_names.get(dataset, dataset)
            ds_color = dataset_colors.get(dataset, '#95a5a6')
            dataset_results = datasets[dataset]
            
            # Sort by filename
            dataset_results = sorted(dataset_results, key=lambda x: x.filename)
            
            successful = len([r for r in dataset_results if not r.failed])
            failed = len([r for r in dataset_results if r.failed])
            
            f.write(f"<div id='{dataset}-section' class='dataset-section dataset-{dataset}'>\n")
            f.write(f"<h2 style='color: {ds_color}; margin-top: 0;'>{ds_name}</h2>\n")
            f.write(f"<p><strong>{len(dataset_results)} experiments</strong> ({successful} successful, {failed} failed)</p>\n")
            
            # Single table with all experiments (filename first)
            f.write("<table>\n")
            f.write("<tr><th>Filename</th><th>Loss</th><th>Best Acc</th><th>Final Acc</th><th>Epochs</th><th>BS</th><th>LR</th><th>Time</th><th>Status</th></tr>\n")
            
            for r in dataset_results:
                status_class = 'failed' if r.failed else 'success'
                status = "❌" if r.failed else "✅"
                best_acc = f"<strong>{r.best_accuracy:.2f}%</strong>" if r.best_accuracy else "N/A"
                final_acc = f"{r.final_accuracy:.2f}%" if r.final_accuracy else "N/A"
                time_str = f"{r.total_time_sec/60:.1f}min" if r.total_time_sec else "N/A"
                lr_str = f"{r.lr}" if r.lr else "N/A"
                bs_str = f"{r.batch_size}" if r.batch_size else "N/A"
                filename = r.filename.replace('.txt', '')
                # Get base loss type (ce, hinge, vlog)
                loss_type = r.loss_type.split('_')[0] if r.loss_type else "N/A"
                
                f.write(f"<tr>")
                f.write(f"<td><code>{filename}</code></td>")
                f.write(f"<td>{loss_type}</td>")
                f.write(f"<td>{best_acc}</td>")
                f.write(f"<td>{final_acc}</td>")
                f.write(f"<td>{r.epochs}</td>")
                f.write(f"<td>{bs_str}</td>")
                f.write(f"<td>{lr_str}</td>")
                f.write(f"<td>{time_str}</td>")
                f.write(f"<td class='{status_class}'>{status}</td>")
                f.write(f"</tr>\n")
            
            f.write("</table>\n")
            f.write("</div>\n")
        
        # ============== TRAINING CURVES ==============
        f.write("<div id='curves-section'>\n")
        f.write("<h2>Training Curves</h2>\n")
        
        for dataset in sorted(datasets.keys()):
            ds_name = dataset_names.get(dataset, dataset)
            ds_color = dataset_colors.get(dataset, '#95a5a6')
            dataset_results = datasets[dataset]
            # Sort by filename
            dataset_results = sorted(dataset_results, key=lambda x: x.filename)
            
            f.write(f"<h3 style='color: {ds_color};'>{ds_name}</h3>\n")
            
            for r in dataset_results:
                if r.plot_file and os.path.exists(r.plot_file):
                    f.write(f"<div class='plot-container'>\n")
                    
                    status = "❌" if r.failed else "✅"
                    acc_str = f"{r.best_accuracy:.2f}%" if r.best_accuracy else f"{r.final_accuracy:.2f}%"
                    filename = r.filename.replace('.txt', '')
                    
                    f.write(f"<h4><code>{filename}</code> | {acc_str} {status}</h4>\n")
                    
                    try:
                        with open(r.plot_file, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                            f.write(f'<img src="data:image/png;base64,{img_data}" alt="{r.experiment_name}">\n')
                    except:
                        f.write(f'<img src="{r.plot_file}" alt="{r.experiment_name}">\n')
                    
                    f.write("</div>\n")
        
        f.write("</div>\n")
        
        f.write("""
        <hr>
        <p style="text-align: center; color: #7f8c8d;">
            <em>Generated by analyze_all_results.py</em>
        </p>
    </div>
</body>
</html>
""")
    
    print(f"✅ HTML report saved to: {output_file}")
    return output_file


def print_console_summary(results):
    """Print summary to console - simple tables grouped by dataset"""
    print("\n" + "="*120)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*120 + "\n")
    
    print(f"Total experiments: {len(results)}\n")
    
    # Group by dataset
    datasets = group_by_dataset(results)
    dataset_names = {'mnist': 'MNIST', 'cifar10': 'CIFAR-10', 'unknown': 'Unknown'}
    
    # Dataset summaries
    for dataset in sorted(datasets.keys()):
        ds_name = dataset_names.get(dataset, dataset)
        dataset_results = datasets[dataset]
        
        # Sort by filename
        dataset_results = sorted(dataset_results, key=lambda x: x.filename)
        
        successful = len([r for r in dataset_results if not r.failed])
        failed = len([r for r in dataset_results if r.failed])
        
        print("="*120)
        print(f"{ds_name} ({len(dataset_results)} experiments: {successful} ✅, {failed} ❌)")
        print("="*120)
        print(f"{'Filename':<55} {'Loss':<6} {'Best Acc':<10} {'Final Acc':<10} {'Epochs':<8} {'Status':<6}")
        print("-" * 120)
        
        for r in dataset_results:
            status = "❌" if r.failed else "✅"
            best_acc = f"{r.best_accuracy:.2f}%" if r.best_accuracy else "N/A"
            final_acc = f"{r.final_accuracy:.2f}%" if r.final_accuracy else "N/A"
            filename = r.filename.replace('.txt', '')[:53]
            # Get base loss type (ce, hinge, vlog)
            loss_type = r.loss_type.split('_')[0] if r.loss_type else "N/A"
            
            print(f"{filename:<55} {loss_type:<6} {best_acc:<10} {final_acc:<10} {r.epochs:<8} {status:<6}")
        
        print()
    
    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze all experimental results')
    parser.add_argument('--results-dir', type=str, default='../results',
                       help='Directory containing result files (default: ../results)')
    parser.add_argument('--output-format', type=str, choices=['markdown', 'html', 'both'], 
                       default='both',
                       help='Output format (default: both)')
    
    args = parser.parse_args()
    
    print("🔍 Analyzing experimental results...\n")
    
    # Collect all results
    results = collect_all_results(args.results_dir)
    
    if not results:
        print("❌ No results found to analyze!")
        return
    
    print(f"✅ Found {len(results)} experiment results\n")
    
    # Print console summary
    print_console_summary(results)
    
    # Generate reports in current directory
    if args.output_format in ['markdown', 'both']:
        md_file = 'RESULTS_ANALYSIS.md'
        generate_markdown_report(results, md_file)
    
    if args.output_format in ['html', 'both']:
        html_file = 'RESULTS_ANALYSIS.html'
        generate_html_report(results, html_file)
    
    print("\n✨ Analysis complete!")
    print("\nGenerated files in summary_results folder:")
    if args.output_format in ['markdown', 'both']:
        print(f"  📝 {md_file}")
    if args.output_format in ['html', 'both']:
        print(f"  🌐 {html_file}")
    print()


if __name__ == '__main__':
    main()

