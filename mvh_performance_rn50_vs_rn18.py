import os 
from matplotlib import pyplot as plt
import csv
import argparse

def main():
    # Set figure size and style
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 42,
        'axes.titlesize': 42,
        'xtick.labelsize': 42,
        'ytick.labelsize': 42,
        'legend.fontsize': 25,
        'figure.figsize': (15, 10),
        'figure.dpi': 300
    })

    base_dir = "/scratch/tesi_magistrale/"
    
    # Lists to store rotation accuracy for both models
    resnet18_rotation = []
    resnet50_rotation = []
    
    # Maximum pruning steps
    max_prune = 26
    
    # Read ResNet18 data
    for step in range(max_prune):
        try:
            with open(base_dir + "resnet18" + f'_pruning_step_{str(step)}.csv', mode='r') as csvfile:
                linereader = list(csv.reader(csvfile))
                # Rotation data is in row 11 (index 10), last column
                resnet18_rotation.append(float(linereader[11][-1]))
        except FileNotFoundError:
            print(f"Warning: Could not find ResNet18 data for pruning step {step}")
    
    # Read ResNet50 data
    for step in range(max_prune):
        try:
            with open(base_dir + "resnet50" + f'_pruning_step_{str(step)}.csv', mode='r') as csvfile:
                linereader = list(csv.reader(csvfile))
                # Rotation data is in row 11 (index 10), last column
                resnet50_rotation.append(float(linereader[11][-1]))
        except FileNotFoundError:
            print(f"Warning: Could not find ResNet50 data for pruning step {step}")
    
    # Create the plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    
    # Plot both models' rotation performance using the same blue color but different line patterns
    plt.plot(resnet18_rotation, label='ResNet18 on rotation', linewidth=10, color='#1f77b4', linestyle='-', marker='s', markersize=6)
    plt.plot(resnet50_rotation, label='ResNet50 on rotation', linewidth=10, color='#1f77b4', linestyle='--', marker='s', markersize=6)
    
    # Set axis limits and labels
    plt.ylim(0, 1)
    plt.xlabel('Pruning step', fontsize=42)
    plt.ylabel('Accuracy', fontsize=42)
    # Remove title to match the provided example
    
    # Add grid for better readability
    plt.grid(True, linestyle='-', alpha=0.1)
    
    # Adjust legend to match the provided example
    plt.legend(loc='upper right', fontsize=42, frameon=True, edgecolor='black')
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_dir = os.path.join(base_dir, 'model_vs_human_performances', 'comparison')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as PDF with high quality
    save_path = os.path.join(save_dir, 'resnet18_vs_resnet50_rotation.pdf')
    plt.savefig(save_path, 
               format='pdf', 
               bbox_inches='tight',
               dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    main()