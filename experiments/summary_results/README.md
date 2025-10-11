# Summary Results

This folder contains the analysis script and generated reports.

**📊 Important**: All plots contain **both Loss AND Accuracy curves** side-by-side:
- Left panel: Loss (train vs test)
- Right panel: Accuracy (train vs test)

## Usage

```bash
# From this folder (summary_results)
python analyze_all_results.py

# Or from experiments folder
python summary_results/analyze_all_results.py
```

## Output

The script generates two files in this folder:
- `RESULTS_ANALYSIS.md` - Markdown report with tables and plot references
- `RESULTS_ANALYSIS.html` - Self-contained HTML report with embedded plots

## Options

```bash
# Generate only markdown
python analyze_all_results.py --output-format markdown

# Generate only HTML
python analyze_all_results.py --output-format html

# Specify custom results directory
python analyze_all_results.py --results-dir /path/to/results
```

## Sharing

- **Email**: Send the HTML file (plots are embedded)
- **GitHub**: Commit the markdown file
- **Presentation**: Open HTML in browser

