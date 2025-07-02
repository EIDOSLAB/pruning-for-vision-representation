import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import itertools

def main(args):
    # Load performance data from model-specific file
    performance_path = os.path.join('/scratch/tesi_magistrale/vit_b_32/performance.txt')
    with open(performance_path, 'r') as f:
        perfs = f.readlines()
    
    # Parse sparsities and accuracies
    sparsities = []
    accuracies = []
    for perf in perfs[1:]:  # Skip header row
        values = perf.rstrip().split('\t')
        accuracies.append(float(values[0]))
        sparsities.append(round(float(values[1]), 1))
    
    if args.model_name == 'resnet50':
        # Limit data points if needed
        sparsities = sparsities[:args.model_max_prune+1]
        accuracies = accuracies[:args.model_max_prune+1]
    
    # Initialize figure dimensions and style settings
    plt.rcParams['figure.figsize'] = [16, 5]  # Set default figure size
    plt.rcParams['figure.dpi'] = 300  # Set default DPI
    plt.rcParams['font.size'] = 32  # Set default font size
    
    make_plot(args, 'gradCAM', sparsities, accuracies)
    make_plot(args, 'attention', sparsities, accuracies)
    make_plot(args, 'ig', sparsities, accuracies)

def make_plot(args, explain_method, sparsities, accuracies):
    if args.model_name == 'vit_b_32':
        methods = ['attention', 'ig']
    else:
        methods = ['gradCAM', 'guided_gradCAM', 'ig']
        
    # Define line styles for different methodologies
    line_styles = {
        'gradCAM': '-',
        'guided_gradCAM': '--',
        'attention': '-.',
        'ig': ':'
    }
    
    # Define display names for methods in legend
    method_names = {
        'gradCAM': 'GradCAM',
        'guided_gradCAM': 'Guided-GradCAM',
        'attention': 'Attention',
        'ig': 'IG'
    }
    
    # Define colors for metrics
    rma_color = '#1f77b4'  # blue
    rra_color = '#d62728'  # red
    acc_color = '#2ca02c'  # green
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Create a second y-axis for accuracy
    ax2 = ax.twinx()
    
    # Set axis labels
    ax.set_xlabel('Sparsity', fontsize=32)
    ax.set_ylabel('Score', fontsize=32)
    ax2.set_ylabel('Accuracy', fontsize=32, color=acc_color)
    
    # Set tick parameters
    ax.tick_params(axis='x', which='major', labelsize=32)
    ax.tick_params(axis='y', which='major', labelsize=32)
    ax2.tick_params(axis='y', which='major', labelsize=32, colors=acc_color)
    
    for explain_method in methods:
        masses = []
        ranks = []
        for pruning_iteration in range(args.model_max_prune+1):
            result_file_path = os.path.join('/scratch/tesi_magistrale', 'output', 
                f'{explain_method}_{args.model_name}_pruning_iteration_{pruning_iteration:02}.txt')
            f = open(result_file_path, 'r')
            masses.append(float(f.readline().split(':')[-1].strip()))
            ranks.append(float(f.readline().split(':')[-1].strip()))
            f.close()
            
        # Plot RMA with blue color and methodology-specific line style
        ax.plot(np.array(sparsities).astype('str'), masses, 
                label=f'RMA {method_names[explain_method]}',
                color=rma_color,
                linestyle=line_styles[explain_method],
                linewidth=3)
        
        # Plot RRA with red color and methodology-specific line style
        ax.plot(np.array(sparsities).astype('str'), ranks, 
                label=f'RRA {method_names[explain_method]}',
                color=rra_color,
                linestyle=line_styles[explain_method],
                linewidth=3)
    
    # Plot accuracy on the secondary y-axis
    ax2.plot(np.array(sparsities).astype('str'), accuracies, 
            label='Accuracy',
            color=acc_color,
            linestyle='-',
            linewidth=3,
            marker='o',
            markersize=6)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Position the legend below the plot
    plt.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', bbox_to_anchor=(0.5, -0.4),
              fancybox=True, shadow=True, ncol=5, fontsize=32).set_visible(True)
    
    # Show only every 2nd x-tick label
    for i, tick in enumerate(ax.get_xticklabels()):
        if i % 2 != 0:  # Hide every other tick
            tick.set_visible(False)
        else:
            tick.set_rotation(30)
    
    # Adjust bottom margin to accommodate legend and x-label
    plt.subplots_adjust(bottom=0.25)
    
    plt.savefig(f'/scratch/tesi_magistrale/output/{args.model_name}_{explain_method}_plot.pdf',
                bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.close()

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--performance-path", default=None, type=str, help="File with model performance in relation to its sparsity")
    parser.add_argument("--root-results-path", default=None, type=str, help="Root directory for results file, corloc test LOST")
    parser.add_argument("--model-name", default=None, type=str, help="Model Name")
    parser.add_argument("--model-max-prune", default=None, type=int, help="Root directory for results file, corloc test LOST")
    parser.add_argument("--model-baseline", default=None, type=float, help="Performance for the chosen model in LOST corloc task")
    parser.add_argument("--dilation", default=2, type=int, help="Performance for the chosen model in LOST corloc task")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)