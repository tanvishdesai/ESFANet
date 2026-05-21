"""
Training Log Visualization Script
=================================
Generates publication-quality graphs and comparison tables from federated learning training logs.

Models: Consensus Only, FedAlert Only, FedAvg, FedProx, Full
Scenarios: A, B, C (different data splitting scenarios)

Output:
- Individual training curves per metric
- Comparison plots across all models/scenarios
- LaTeX and Markdown comparison tables
"""

import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =====================================
# Configuration
# =====================================

# Path to training logs directory
LOGS_DIR = Path(__file__).parent / "training logs"

# Output directory for plots and tables
OUTPUT_DIR = Path(__file__).parent / "plots"

# Model name mapping (log file prefix -> display name)
MODEL_MAPPING = {
    'consensus only': 'Consensus Only',
    'consensusonly': 'Consensus Only',
    'fedalert only': 'FedAlert Only',
    'fedalert-only': 'FedAlert Only',
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx',
    'full': 'Full (FedAlert + Consensus)'
}

# Scenario mapping
SCENARIOS = ['a', 'b', 'c']

# Metrics to extract
METRICS = ['Acc', 'Prec', 'Rec', 'F1', 'AUC']
METRIC_DISPLAY_NAMES = {
    'Acc': 'Accuracy',
    'Prec': 'Precision', 
    'Rec': 'Recall',
    'F1': 'F1-Score',
    'AUC': 'AUC-ROC'
}

# Color palette for models (research paper friendly - colorblind safe)
MODEL_COLORS = {
    'Consensus Only': '#1f77b4',      # Blue
    'FedAlert Only': '#ff7f0e',        # Orange
    'FedAvg': '#2ca02c',               # Green
    'FedProx': '#d62728',              # Red
    'Full (FedAlert + Consensus)': '#9467bd'  # Purple
}

# Line styles for scenarios
SCENARIO_STYLES = {
    'a': '-',    # Solid
    'b': '--',   # Dashed
    'c': ':'     # Dotted
}

# Marker styles for models
MODEL_MARKERS = {
    'Consensus Only': 'o',
    'FedAlert Only': 's',
    'FedAvg': '^',
    'FedProx': 'D',
    'Full (FedAlert + Consensus)': 'p'
}

# =====================================
# Log Parsing Functions
# =====================================

def parse_log_file(filepath):
    """
    Parse a training log file and extract round-by-round metrics.
    
    Returns:
        dict: Contains 'rounds' dict with metric arrays, 'final_metrics' dict,
              and 'final_metrics_fedalert' dict (at threshold 0.75)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {
        'rounds': defaultdict(list),
        'final_metrics': {},
        'final_metrics_fedalert': {}
    }
    
    # Pattern for round results: Round N Results (Thresh=0.50): Acc: X, Prec: X, ...
    round_pattern = r'Round\s+(\d+)\s+Results\s+\(Thresh=0\.50\):\s+Acc:\s+([\d.]+),\s+Prec:\s+([\d.]+),\s+Rec:\s+([\d.]+),\s+F1:\s+([\d.]+),\s+AUC:\s+([\d.]+)'
    
    for match in re.finditer(round_pattern, content):
        round_num = int(match.group(1))
        results['rounds']['round'].append(round_num)
        results['rounds']['Acc'].append(float(match.group(2)))
        results['rounds']['Prec'].append(float(match.group(3)))
        results['rounds']['Rec'].append(float(match.group(4)))
        results['rounds']['F1'].append(float(match.group(5)))
        results['rounds']['AUC'].append(float(match.group(6)))
    
    # Pattern for final results at 0.50 threshold
    final_pattern_050 = r'Classification Performance at Standard Threshold \(0\.50\):\s*\n\s*Final Accuracy:\s+([\d.]+)\s*\n\s*Final Precision:\s+([\d.]+)\s*\n\s*Final Recall:\s+([\d.]+)\s*\n\s*Final F1-Score:\s+([\d.]+)\s*\n\s*Final AUC:\s+([\d.]+)'
    
    match = re.search(final_pattern_050, content)
    if match:
        results['final_metrics'] = {
            'Accuracy': float(match.group(1)),
            'Precision': float(match.group(2)),
            'Recall': float(match.group(3)),
            'F1-Score': float(match.group(4)),
            'AUC': float(match.group(5))
        }
    
    # Pattern for final results at 0.75 threshold (FedAlert)
    final_pattern_075 = r'Classification Performance at FedAlert Threshold \(0\.75\):\s*\n\s*Final Accuracy:\s+([\d.]+)\s*\n\s*Final Precision:\s+([\d.]+)\s*\n\s*Final Recall:\s+([\d.]+)\s*\n\s*Final F1-Score:\s+([\d.]+)\s*\n\s*Final AUC:\s+([\d.]+)'
    
    match = re.search(final_pattern_075, content)
    if match:
        results['final_metrics_fedalert'] = {
            'Accuracy': float(match.group(1)),
            'Precision': float(match.group(2)),
            'Recall': float(match.group(3)),
            'F1-Score': float(match.group(4)),
            'AUC': float(match.group(5))
        }
    
    return results


def identify_model_scenario(filename):
    """
    Identify the model and scenario from a log filename.
    
    Returns:
        tuple: (model_display_name, scenario_letter) or (None, None) if not recognized
    """
    filename_lower = filename.lower().replace('.txt', '').strip()
    
    # Extract scenario (last character before .txt)
    scenario = None
    for s in SCENARIOS:
        if filename_lower.endswith(f' {s}') or filename_lower.endswith(f'-{s}') or filename_lower.endswith(s):
            scenario = s
            break
    
    if not scenario:
        return None, None
    
    # Extract model name
    model_display = None
    for pattern, display_name in MODEL_MAPPING.items():
        if pattern in filename_lower:
            model_display = display_name
            break
    
    return model_display, scenario


def load_all_logs(logs_dir):
    """
    Load all training logs from the specified directory.
    
    Returns:
        dict: Nested dict {model: {scenario: parsed_data}}
    """
    all_data = defaultdict(dict)
    
    for log_file in logs_dir.glob('*.txt'):
        model, scenario = identify_model_scenario(log_file.name)
        
        if model and scenario:
            print(f"Loading: {log_file.name} -> Model: {model}, Scenario: {scenario.upper()}")
            all_data[model][scenario] = parse_log_file(log_file)
        else:
            print(f"Warning: Could not identify model/scenario for {log_file.name}")
    
    return dict(all_data)


# =====================================
# Plotting Functions
# =====================================

def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2,
        'lines.markersize': 5,
    })


def plot_metric_by_model_all_scenarios(all_data, metric, output_dir):
    """
    Create a single plot showing a metric across all models, with different line styles for scenarios.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in MODEL_COLORS.keys():
        if model not in all_data:
            continue
        
        for scenario in SCENARIOS:
            if scenario not in all_data[model]:
                continue
            
            data = all_data[model][scenario]
            rounds = data['rounds']['round']
            values = data['rounds'][metric]
            
            label = f"{model} (Scenario {scenario.upper()})"
            ax.plot(rounds, values, 
                   color=MODEL_COLORS[model],
                   linestyle=SCENARIO_STYLES[scenario],
                   marker=MODEL_MARKERS[model],
                   markevery=3,
                   label=label,
                   alpha=0.9)
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel(METRIC_DISPLAY_NAMES[metric])
    ax.set_title(f'{METRIC_DISPLAY_NAMES[metric]} vs. Communication Rounds')
    ax.set_xlim(1, 20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True)
    
    plt.tight_layout()
    save_path = output_dir / f'{metric.lower()}_all_models_scenarios.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{metric.lower()}_all_models_scenarios.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metric_comparison_per_scenario(all_data, metric, output_dir):
    """
    Create a subplot grid with one plot per scenario, comparing all models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    
    for idx, scenario in enumerate(SCENARIOS):
        ax = axes[idx]
        
        for model in MODEL_COLORS.keys():
            if model not in all_data or scenario not in all_data[model]:
                continue
            
            data = all_data[model][scenario]
            rounds = data['rounds']['round']
            values = data['rounds'][metric]
            
            ax.plot(rounds, values,
                   color=MODEL_COLORS[model],
                   marker=MODEL_MARKERS[model],
                   markevery=3,
                   label=model,
                   linewidth=2,
                   alpha=0.9)
        
        ax.set_xlabel('Communication Round')
        if idx == 0:
            ax.set_ylabel(METRIC_DISPLAY_NAMES[metric])
        ax.set_title(f'Scenario {scenario.upper()}')
        ax.set_xlim(1, 20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        ax.grid(True, alpha=0.3)
    
    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.08), frameon=True)
    
    plt.suptitle(f'{METRIC_DISPLAY_NAMES[metric]} Comparison Across Scenarios', y=1.12, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / f'{metric.lower()}_by_scenario.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{metric.lower()}_by_scenario.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_all_metrics_single_model(all_data, model, output_dir):
    """
    Create a 2x3 grid showing all metrics for a single model across scenarios.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        
        for scenario in SCENARIOS:
            if scenario not in all_data.get(model, {}):
                continue
            
            data = all_data[model][scenario]
            rounds = data['rounds']['round']
            values = data['rounds'][metric]
            
            ax.plot(rounds, values,
                   linestyle=SCENARIO_STYLES[scenario],
                   marker='o',
                   markevery=4,
                   label=f'Scenario {scenario.upper()}',
                   linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel(METRIC_DISPLAY_NAMES[metric])
        ax.set_title(METRIC_DISPLAY_NAMES[metric])
        ax.set_xlim(1, 20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Remove the 6th subplot (we only have 5 metrics)
    axes[5].axis('off')
    
    plt.suptitle(f'{model} - Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    model_filename = model.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'and')
    save_path = output_dir / f'{model_filename}_all_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{model_filename}_all_metrics.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_f1_auc_combined(all_data, output_dir):
    """
    Create a combined F1-Score and AUC comparison plot (most important metrics for research).
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Row 1: F1-Score for each scenario
    # Row 2: AUC for each scenario
    
    for row, metric in enumerate(['F1', 'AUC']):
        for col, scenario in enumerate(SCENARIOS):
            ax = axes[row, col]
            
            for model in MODEL_COLORS.keys():
                if model not in all_data or scenario not in all_data[model]:
                    continue
                
                data = all_data[model][scenario]
                rounds = data['rounds']['round']
                values = data['rounds'][metric]
                
                ax.plot(rounds, values,
                       color=MODEL_COLORS[model],
                       marker=MODEL_MARKERS[model],
                       markevery=3,
                       label=model,
                       linewidth=2,
                       alpha=0.9)
            
            ax.set_xlabel('Round')
            if col == 0:
                ax.set_ylabel(METRIC_DISPLAY_NAMES[metric])
            ax.set_title(f'{METRIC_DISPLAY_NAMES[metric]} - Scenario {scenario.upper()}')
            ax.set_xlim(1, 20)
            ax.grid(True, alpha=0.3)
    
    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.02), frameon=True)
    
    plt.suptitle('F1-Score and AUC Comparison', y=1.06, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'f1_auc_combined.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'f1_auc_combined.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_metrics_bar_chart(all_data, output_dir):
    """
    Create a bar chart comparing final metrics across all models and scenarios.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    model_names = list(MODEL_COLORS.keys())
    x = np.arange(len(model_names))
    width = 0.25
    
    for idx, scenario in enumerate(SCENARIOS):
        ax = axes[idx]
        
        metrics_to_plot = ['Accuracy', 'F1-Score', 'AUC']
        
        for i, metric in enumerate(metrics_to_plot):
            values = []
            for model in model_names:
                if model in all_data and scenario in all_data[model]:
                    values.append(all_data[model][scenario]['final_metrics'].get(metric, 0))
                else:
                    values.append(0)
            
            bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.85)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title(f'Scenario {scenario.upper()} - Final Performance')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace(' (FedAlert + Consensus)', '\n(Full)') for m in model_names], 
                          rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0.7, 1.0)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Final Performance Comparison (Threshold = 0.50)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'final_metrics_bar_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'final_metrics_bar_chart.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =====================================
# Table Generation Functions
# =====================================

def generate_comparison_table_markdown(all_data, output_dir):
    """
    Generate a Markdown comparison table of final metrics.
    """
    lines = []
    lines.append("# Final Performance Comparison Table\n")
    lines.append("## Standard Threshold (0.50)\n")
    
    # Header
    header = "| Model | Scenario | Accuracy | Precision | Recall | F1-Score | AUC |"
    separator = "|-------|----------|----------|-----------|--------|----------|-----|"
    lines.append(header)
    lines.append(separator)
    
    for model in MODEL_COLORS.keys():
        if model not in all_data:
            continue
        for scenario in SCENARIOS:
            if scenario not in all_data[model]:
                continue
            
            metrics = all_data[model][scenario]['final_metrics']
            row = f"| {model} | {scenario.upper()} | {metrics['Accuracy']:.4f} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} | {metrics['F1-Score']:.4f} | {metrics['AUC']:.4f} |"
            lines.append(row)
    
    lines.append("\n## FedAlert Threshold (0.75)\n")
    lines.append(header)
    lines.append(separator)
    
    for model in MODEL_COLORS.keys():
        if model not in all_data:
            continue
        for scenario in SCENARIOS:
            if scenario not in all_data[model]:
                continue
            
            metrics = all_data[model][scenario].get('final_metrics_fedalert', {})
            if metrics:
                row = f"| {model} | {scenario.upper()} | {metrics['Accuracy']:.4f} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} | {metrics['F1-Score']:.4f} | {metrics['AUC']:.4f} |"
                lines.append(row)
    
    # Add best model summary
    lines.append("\n## Best Performing Models\n")
    
    for scenario in SCENARIOS:
        best_f1 = 0
        best_model = None
        
        for model in all_data:
            if scenario in all_data[model]:
                f1 = all_data[model][scenario]['final_metrics'].get('F1-Score', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
        
        if best_model:
            lines.append(f"- **Scenario {scenario.upper()}**: {best_model} (F1-Score: {best_f1:.4f})")
    
    content = '\n'.join(lines)
    
    save_path = output_dir / 'comparison_table.md'
    with open(save_path, 'w') as f:
        f.write(content)
    print(f"Saved: {save_path}")
    
    return content


def generate_latex_table(all_data, output_dir):
    """
    Generate a LaTeX table for direct inclusion in research papers.
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Final Performance Comparison at Standard Threshold (0.50)}")
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Scenario} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{AUC} \\")
    lines.append(r"\midrule")
    
    # Find best values for each metric and scenario (for bold highlighting)
    best_values = {scenario: {metric: 0 for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']} for scenario in SCENARIOS}
    
    for scenario in SCENARIOS:
        for model in all_data:
            if scenario in all_data[model]:
                metrics = all_data[model][scenario]['final_metrics']
                for metric in best_values[scenario]:
                    if metrics.get(metric, 0) > best_values[scenario][metric]:
                        best_values[scenario][metric] = metrics[metric]
    
    for model in MODEL_COLORS.keys():
        if model not in all_data:
            continue
        for scenario in SCENARIOS:
            if scenario not in all_data[model]:
                continue
            
            metrics = all_data[model][scenario]['final_metrics']
            
            # Format values, bold if best
            values = []
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
                val = metrics.get(metric, 0)
                if abs(val - best_values[scenario][metric]) < 0.0001:
                    values.append(f"\\textbf{{{val:.4f}}}")
                else:
                    values.append(f"{val:.4f}")
            
            model_escaped = model.replace('&', r'\&')
            row = f"{model_escaped} & {scenario.upper()} & {' & '.join(values)} \\\\"
            lines.append(row)
        
        # Add horizontal line between models
        lines.append(r"\midrule")
    
    # Remove the last midrule and add bottomrule
    lines[-1] = r"\bottomrule"
    
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    
    content = '\n'.join(lines)
    
    save_path = output_dir / 'comparison_table.tex'
    with open(save_path, 'w') as f:
        f.write(content)
    print(f"Saved: {save_path}")
    
    return content


def generate_csv_table(all_data, output_dir):
    """
    Generate a CSV file for easy data manipulation.
    """
    lines = []
    lines.append("Model,Scenario,Accuracy,Precision,Recall,F1-Score,AUC,Threshold")
    
    # Standard threshold
    for model in MODEL_COLORS.keys():
        if model not in all_data:
            continue
        for scenario in SCENARIOS:
            if scenario not in all_data[model]:
                continue
            
            metrics = all_data[model][scenario]['final_metrics']
            row = f"{model},{scenario.upper()},{metrics['Accuracy']:.4f},{metrics['Precision']:.4f},{metrics['Recall']:.4f},{metrics['F1-Score']:.4f},{metrics['AUC']:.4f},0.50"
            lines.append(row)
    
    # FedAlert threshold
    for model in MODEL_COLORS.keys():
        if model not in all_data:
            continue
        for scenario in SCENARIOS:
            if scenario not in all_data[model]:
                continue
            
            metrics = all_data[model][scenario].get('final_metrics_fedalert', {})
            if metrics:
                row = f"{model},{scenario.upper()},{metrics['Accuracy']:.4f},{metrics['Precision']:.4f},{metrics['Recall']:.4f},{metrics['F1-Score']:.4f},{metrics['AUC']:.4f},0.75"
                lines.append(row)
    
    content = '\n'.join(lines)
    
    save_path = output_dir / 'comparison_data.csv'
    with open(save_path, 'w') as f:
        f.write(content)
    print(f"Saved: {save_path}")


# =====================================
# Main Execution
# =====================================

def main():
    """Main function to generate all plots and tables."""
    print("=" * 60)
    print("Training Log Visualization Script")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load all logs
    print("\n--- Loading Training Logs ---")
    all_data = load_all_logs(LOGS_DIR)
    
    if not all_data:
        print("ERROR: No training logs found!")
        return
    
    print(f"\nLoaded {sum(len(scenarios) for scenarios in all_data.values())} log files")
    print(f"Models: {list(all_data.keys())}")
    
    # Setup plot style
    setup_plot_style()
    
    # Generate plots
    print("\n--- Generating Plots ---")
    
    # 1. Per-metric comparison across all models and scenarios
    for metric in METRICS:
        plot_metric_by_model_all_scenarios(all_data, metric, OUTPUT_DIR)
        plot_metric_comparison_per_scenario(all_data, metric, OUTPUT_DIR)
    
    # 2. Per-model summary plots
    for model in all_data.keys():
        plot_all_metrics_single_model(all_data, model, OUTPUT_DIR)
    
    # 3. Combined F1 + AUC plot (highlight plot for papers)
    plot_f1_auc_combined(all_data, OUTPUT_DIR)
    
    # 4. Bar charts for final metrics
    plot_final_metrics_bar_chart(all_data, OUTPUT_DIR)
    
    # Generate tables
    print("\n--- Generating Tables ---")
    generate_comparison_table_markdown(all_data, OUTPUT_DIR)
    generate_latex_table(all_data, OUTPUT_DIR)
    generate_csv_table(all_data, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("COMPLETE! All plots and tables generated.")
    print(f"Check the output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
