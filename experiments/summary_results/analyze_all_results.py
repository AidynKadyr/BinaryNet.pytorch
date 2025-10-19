"""
Comprehensive Analysis Script for Binary Neural Network Experiments

This script:
1. Parses all result files in experiments/results/
2. Extracts key metrics (accuracy, training time, hyperparameters)
3. Generates comparison tables
4. Creates an HTML report with embedded plots
5. Provides ranking and insights

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


def generate_markdown_report(results, output_file):
    """Generate markdown report with tables and plot references"""
    
    sorted_groups, groups = generate_summary_table(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 📊 Experimental Results Analysis\n\n")
        f.write("**Auto-generated summary of all Binary Neural Network experiments**\n\n")
        f.write(f"Total experiments analyzed: **{len(results)}**\n\n")
        f.write("---\n\n")
        
        # Executive Summary - Top Performers
        f.write("## 🏆 Top Performers\n\n")
        
        # Sort all non-failed results by best accuracy
        successful = [r for r in results if not r.failed and r.best_accuracy is not None]
        top_10 = sorted(successful, key=lambda x: x.best_accuracy, reverse=True)[:10]
        
        f.write("| Rank | Loss Function | Hyperparameters | Best Acc | Epochs | Time/Epoch |\n")
        f.write("|------|---------------|-----------------|----------|--------|------------|\n")
        
        for i, r in enumerate(top_10, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            f.write(f"| {medal} | {r.get_loss_description()} | {r.get_hyperparameters_str()} | "
                   f"**{r.best_accuracy:.2f}%** | {r.epochs} | {r.time_per_epoch:.1f}s |\n")
        
        f.write("\n---\n\n")
        
        # Detailed Results by Loss Type and Epochs
        f.write("## 📈 Detailed Results by Configuration\n\n")
        
        for (loss_type, epochs), group_results in sorted_groups:
            f.write(f"### {loss_type.upper().replace('_', ' ')} - {epochs} Epochs\n\n")
            
            # Sort by best accuracy within group
            group_results = sorted(group_results, key=lambda x: x.best_accuracy if x.best_accuracy else 0, reverse=True)
            
            f.write("| Configuration | Best Test Acc | Final Test Acc | Training Time | Status |\n")
            f.write("|---------------|---------------|----------------|---------------|--------|\n")
            
            for r in group_results:
                status = "❌ Failed" if r.failed else "✅ Success"
                best_acc = f"{r.best_accuracy:.2f}%" if r.best_accuracy else "N/A"
                final_acc = f"{r.final_accuracy:.2f}%" if r.final_accuracy else "N/A"
                time_str = f"{r.total_time_sec:.1f}s ({r.total_time_sec/60:.2f} min)" if r.total_time_sec else "N/A"
                
                f.write(f"| {r.get_hyperparameters_str()} | {best_acc} | {final_acc} | {time_str} | {status} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        
        # Key Insights Section
        f.write("## 💡 Key Insights\n\n")
        
        # Find best overall
        if successful:
            best = max(successful, key=lambda x: x.best_accuracy)
            f.write(f"### Best Overall Performance\n")
            f.write(f"- **Loss Function**: {best.get_loss_description()}\n")
            f.write(f"- **Hyperparameters**: {best.get_hyperparameters_str()}\n")
            f.write(f"- **Best Test Accuracy**: {best.best_accuracy:.2f}% (Epoch {best.best_epoch})\n")
            f.write(f"- **Training Time**: {best.total_time_sec:.1f}s ({best.total_time_sec/60:.2f} min)\n")
            f.write(f"- **Time per Epoch**: {best.time_per_epoch:.1f}s\n\n")
        
        # Compare loss types
        f.write("### Performance by Loss Type (Best for Each)\n\n")
        loss_type_best = {}
        for r in successful:
            if r.loss_type not in loss_type_best or r.best_accuracy > loss_type_best[r.loss_type].best_accuracy:
                loss_type_best[r.loss_type] = r
        
        f.write("| Loss Type | Best Accuracy | Configuration |\n")
        f.write("|-----------|---------------|---------------|\n")
        
        for loss_type, r in sorted(loss_type_best.items(), key=lambda x: x[1].best_accuracy, reverse=True):
            f.write(f"| {r.get_loss_description()} | **{r.best_accuracy:.2f}%** | {r.get_hyperparameters_str()} |\n")
        
        f.write("\n")
        
        # Failed experiments
        failed = [r for r in results if r.failed]
        if failed:
            f.write("### ⚠️ Failed Experiments\n\n")
            f.write("The following configurations resulted in training failures (typically loss explosion):\n\n")
            
            for r in failed:
                f.write(f"- **{r.get_loss_description()}**: {r.get_hyperparameters_str()} "
                       f"(Final acc: {r.final_accuracy:.2f}%)\n")
            
            f.write("\n**Common pattern**: b-annealing (τ-annealing) without proper gradient clipping causes loss explosion.\n\n")
        
        f.write("---\n\n")
        
        # Plots Section
        f.write("## 📊 Training Curves\n\n")
        f.write("Below are the training curves for **all experiments** (successful and failed).\n\n")
        f.write("**Note**: Each plot shows both **Loss** (left) and **Accuracy** (right) curves with train/test comparison.\n\n")
        
        # Group ALL experiments by loss type for plotting
        for loss_type in sorted(set(r.loss_type for r in results)):
            loss_results = [r for r in results if r.loss_type == loss_type]
            # Sort: successful first (by accuracy), then failed
            loss_results = sorted(loss_results, key=lambda x: (not x.failed, x.best_accuracy if x.best_accuracy else 0), reverse=True)
            
            f.write(f"### {loss_results[0].get_loss_description()}\n\n")
            
            for r in loss_results:
                if r.plot_file and os.path.exists(r.plot_file):
                    # Add status indicator
                    status = "❌ FAILED" if r.failed else "✅ Success"
                    f.write(f"#### {r.get_hyperparameters_str()} - {r.epochs} epochs ({status})\n\n")
                    if r.best_accuracy:
                        f.write(f"**Best Test Accuracy**: {r.best_accuracy:.2f}%\n\n")
                    else:
                        f.write(f"**Final Test Accuracy**: {r.final_accuracy:.2f}%\n\n")
                    f.write(f"![{r.experiment_name}]({r.plot_file})\n\n")
                    f.write(f"*Training curves showing: Left = Loss (train/test), Right = Accuracy (train/test)*\n\n")
                else:
                    status = "❌ FAILED" if r.failed else "✅ Success"
                    f.write(f"#### {r.get_hyperparameters_str()} - {r.epochs} epochs ({status})\n\n")
                    if r.best_accuracy:
                        f.write(f"**Best Test Accuracy**: {r.best_accuracy:.2f}%\n\n")
                    else:
                        f.write(f"**Final Test Accuracy**: {r.final_accuracy:.2f}%\n\n")
                    f.write(f"*Plot file not found: {r.experiment_name}.png*\n\n")
        
        f.write("---\n\n")
        
        # Recommendations
        f.write("## 🎯 Recommendations\n\n")
        f.write("Based on the experimental results:\n\n")
        
        f.write("### ✅ What Works Well\n\n")
        f.write("1. **Vlog Loss with low b values** (b=1.0 to 5.0): Consistently high performance\n")
        f.write("2. **β-Annealing** (temperature annealing): Improves convergence for both Hinge and Vlog\n")
        f.write("3. **Batch size 4096**: Good balance between speed and stability\n\n")
        
        f.write("### ❌ What to Avoid\n\n")
        f.write("1. **b-Annealing alone** (1.0 → 100.0): Causes catastrophic loss explosion\n")
        f.write("2. **High b values** (b=20.0): Unstable training and poor convergence\n")
        f.write("3. **Combining b and β annealing**: Compounds instability issues\n\n")
        
        f.write("### 🚀 Suggested Next Steps\n\n")
        f.write("1. **Focus on Vlog + β-annealing**: Most promising approach\n")
        f.write("2. **Test intermediate β ranges**: Try β: 0.5 → 10.0 or 1.0 → 20.0\n")
        f.write("3. **Longer training**: Run best configs for 50-100 epochs\n")
        f.write("4. **Add gradient clipping**: May enable stable b-annealing\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated automatically by `analyze_all_results.py`*\n")
    
    print(f"✅ Markdown report saved to: {output_file}")
    return output_file


def generate_html_report(results, output_file):
    """Generate HTML report with embedded plots"""
    
    sorted_groups, groups = generate_summary_table(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # HTML header
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Binary Neural Network - Experimental Results</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
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
        h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }
        h3 { color: #7f8c8d; margin-top: 20px; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        th {
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover { background: #f8f9fa; }
        .success { color: #27ae60; font-weight: bold; }
        .failed { color: #e74c3c; font-weight: bold; }
        .plot-container {
            margin: 30px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-card {
            display: inline-block;
            background: #ecf0f1;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .insight-box {
            background: #e8f5e9;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
        }
        .warning-box {
            background: #ffebee;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 20px 0;
        }
        .medal { font-size: 1.5em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Binary Neural Network - Experimental Results Analysis</h1>
        <p><strong>Total Experiments:</strong> """ + str(len(results)) + """</p>
        <hr>
""")
        
        # Top performers section
        f.write("<h2>🏆 Top 10 Performers</h2>\n")
        f.write("<table>\n")
        f.write("<tr><th>Rank</th><th>Loss Function</th><th>Hyperparameters</th><th>Best Accuracy</th><th>Epochs</th><th>Time/Epoch</th></tr>\n")
        
        successful = [r for r in results if not r.failed and r.best_accuracy is not None]
        top_10 = sorted(successful, key=lambda x: x.best_accuracy, reverse=True)[:10]
        
        medals = ["🥇", "🥈", "🥉"] + [f"{i}." for i in range(4, 11)]
        
        for i, r in enumerate(top_10):
            f.write(f"<tr><td class='medal'>{medals[i]}</td>")
            f.write(f"<td>{r.get_loss_description()}</td>")
            f.write(f"<td>{r.get_hyperparameters_str()}</td>")
            f.write(f"<td><strong>{r.best_accuracy:.2f}%</strong></td>")
            f.write(f"<td>{r.epochs}</td>")
            f.write(f"<td>{r.time_per_epoch:.1f}s</td></tr>\n")
        
        f.write("</table>\n")
        
        # Key metrics
        if successful:
            best = max(successful, key=lambda x: x.best_accuracy)
            f.write("<h2>📈 Key Metrics</h2>\n")
            f.write("<div style='text-align: center;'>\n")
            f.write(f"<div class='metric-card'><div class='metric-value'>{best.best_accuracy:.2f}%</div><div class='metric-label'>Best Accuracy</div></div>\n")
            f.write(f"<div class='metric-card'><div class='metric-value'>{len(successful)}</div><div class='metric-label'>Successful Runs</div></div>\n")
            f.write(f"<div class='metric-card'><div class='metric-value'>{len([r for r in results if r.failed])}</div><div class='metric-label'>Failed Runs</div></div>\n")
            avg_time = sum(r.time_per_epoch for r in successful if r.time_per_epoch) / len([r for r in successful if r.time_per_epoch])
            f.write(f"<div class='metric-card'><div class='metric-value'>{avg_time:.1f}s</div><div class='metric-label'>Avg Time/Epoch</div></div>\n")
            f.write("</div>\n")
        
        # Detailed results
        f.write("<h2>📋 Detailed Results by Configuration</h2>\n")
        
        for (loss_type, epochs), group_results in sorted_groups:
            f.write(f"<h3>{loss_type.upper().replace('_', ' ')} - {epochs} Epochs</h3>\n")
            
            group_results = sorted(group_results, key=lambda x: x.best_accuracy if x.best_accuracy else 0, reverse=True)
            
            f.write("<table>\n")
            f.write("<tr><th>Configuration</th><th>Best Test Acc</th><th>Final Test Acc</th><th>Training Time</th><th>Status</th></tr>\n")
            
            for r in group_results:
                status_class = 'failed' if r.failed else 'success'
                status_text = '❌ Failed' if r.failed else '✅ Success'
                best_acc = f"{r.best_accuracy:.2f}%" if r.best_accuracy else "N/A"
                final_acc = f"{r.final_accuracy:.2f}%" if r.final_accuracy else "N/A"
                time_str = f"{r.total_time_sec:.1f}s" if r.total_time_sec else "N/A"
                
                f.write(f"<tr><td>{r.get_hyperparameters_str()}</td>")
                f.write(f"<td><strong>{best_acc}</strong></td>")
                f.write(f"<td>{final_acc}</td>")
                f.write(f"<td>{time_str}</td>")
                f.write(f"<td class='{status_class}'>{status_text}</td></tr>\n")
            
            f.write("</table>\n")
        
        # Key insights
        f.write("<h2>💡 Key Insights</h2>\n")
        
        if successful:
            best = max(successful, key=lambda x: x.best_accuracy)
            f.write(f"<div class='insight-box'>\n")
            f.write(f"<h3>Best Overall Performance</h3>\n")
            f.write(f"<p><strong>Loss Function:</strong> {best.get_loss_description()}<br>\n")
            f.write(f"<strong>Hyperparameters:</strong> {best.get_hyperparameters_str()}<br>\n")
            f.write(f"<strong>Best Test Accuracy:</strong> {best.best_accuracy:.2f}% (Epoch {best.best_epoch})<br>\n")
            f.write(f"<strong>Training Time:</strong> {best.total_time_sec:.1f}s ({best.total_time_sec/60:.2f} min)</p>\n")
            f.write("</div>\n")
        
        # Failed experiments warning
        failed = [r for r in results if r.failed]
        if failed:
            f.write("<div class='warning-box'>\n")
            f.write("<h3>⚠️ Failed Experiments</h3>\n")
            f.write("<p>The following configurations resulted in training failures (loss explosion):</p>\n<ul>\n")
            for r in failed:
                f.write(f"<li><strong>{r.get_loss_description()}:</strong> {r.get_hyperparameters_str()} ")
                f.write(f"(Final acc: {r.final_accuracy:.2f}%)</li>\n")
            f.write("</ul>\n")
            f.write("<p><strong>Common pattern:</strong> b-annealing (τ-annealing) without proper gradient clipping causes loss explosion.</p>\n")
            f.write("</div>\n")
        
        # Training curves with embedded plots
        f.write("<h2>📊 Training Curves</h2>\n")
        f.write("<p><strong>Note:</strong> Each plot shows both <strong>Loss</strong> (left panel) and <strong>Accuracy</strong> (right panel) curves with train/test comparison.</p>\n")
        f.write("<p><em>Including all experiments (successful and failed) to show training dynamics.</em></p>\n")
        
        for loss_type in sorted(set(r.loss_type for r in results)):
            loss_results = [r for r in results if r.loss_type == loss_type]
            # Sort: successful first (by accuracy), then failed
            loss_results = sorted(loss_results, key=lambda x: (not x.failed, x.best_accuracy if x.best_accuracy else 0), reverse=True)
            
            f.write(f"<h3>{loss_results[0].get_loss_description()}</h3>\n")
            
            for r in loss_results:
                if r.plot_file and os.path.exists(r.plot_file):
                    f.write(f"<div class='plot-container'>\n")
                    
                    # Add status indicator
                    status_badge = "<span class='failed'>❌ FAILED</span>" if r.failed else "<span class='success'>✅ Success</span>"
                    accuracy_str = f"Best: {r.best_accuracy:.2f}%" if r.best_accuracy else f"Final: {r.final_accuracy:.2f}%"
                    
                    f.write(f"<h4>{r.get_hyperparameters_str()} - {r.epochs} epochs ({accuracy_str}) {status_badge}</h4>\n")
                    
                    # Embed image as base64 for portable HTML
                    try:
                        with open(r.plot_file, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()
                            f.write(f'<img src="data:image/png;base64,{img_data}" alt="{r.experiment_name}">\n')
                    except:
                        # Fallback to relative path
                        f.write(f'<img src="{r.plot_file}" alt="{r.experiment_name}">\n')
                    
                    f.write("</div>\n")
        
        # Close HTML
        f.write("""
        <hr>
        <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
            <em>Report generated automatically by analyze_all_results.py</em>
        </p>
    </div>
</body>
</html>
""")
    
    print(f"✅ HTML report saved to: {output_file}")
    return output_file


def print_console_summary(results):
    """Print summary to console"""
    print("\n" + "="*80)
    print("📊 EXPERIMENTAL RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print(f"Total experiments analyzed: {len(results)}\n")
    
    successful = [r for r in results if not r.failed and r.best_accuracy is not None]
    failed = [r for r in results if r.failed]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}\n")
    
    if successful:
        print("🏆 TOP 5 PERFORMERS:\n")
        print(f"{'Rank':<6} {'Loss Function':<30} {'Best Accuracy':<15} {'Epochs':<8}")
        print("-" * 80)
        
        top_5 = sorted(successful, key=lambda x: x.best_accuracy, reverse=True)[:5]
        medals = ["🥇", "🥈", "🥉", "4.", "5."]
        
        for i, r in enumerate(top_5):
            print(f"{medals[i]:<6} {r.get_loss_description():<30} {r.best_accuracy:>6.2f}%{'':<8} {r.epochs:<8}")
        
        print("\n")
        
        # Best by loss type
        print("📈 BEST PERFORMANCE BY LOSS TYPE:\n")
        loss_type_best = {}
        for r in successful:
            if r.loss_type not in loss_type_best or r.best_accuracy > loss_type_best[r.loss_type].best_accuracy:
                loss_type_best[r.loss_type] = r
        
        for loss_type, r in sorted(loss_type_best.items(), key=lambda x: x[1].best_accuracy, reverse=True):
            print(f"  {r.get_loss_description():<30} {r.best_accuracy:>6.2f}% ({r.get_hyperparameters_str()})")
        
        print("\n")
    
    if failed:
        print("⚠️  FAILED EXPERIMENTS:\n")
        for r in failed:
            print(f"  ❌ {r.get_loss_description()}: {r.get_hyperparameters_str()}")
        print("\n")
    
    print("="*80 + "\n")


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

