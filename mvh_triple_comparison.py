import os 
from matplotlib import pyplot as plt
import csv

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
    
    # Lists to store accuracy for multiple tasks
    silhouette_accuracy = []
    colour_accuracy = []
    false_colour_accuracy = []
    
    # Maximum pruning steps
    max_prune = 26
    
    # Read ResNet18 data
    for step in range(max_prune):
        try:
            with open(base_dir + "resnet18" + f'_pruning_step_{str(step)}.csv', mode='r') as csvfile:
                linereader = list(csv.reader(csvfile))
                # Row indices based on original script:
                # silhouette_accuracy_top_1 is at index 2
                # colour_accuracy_top_1 is at index 4
                # false_colour_accuracy_top_1 is at index 10
                silhouette_accuracy.append(float(linereader[2][-1]))
                colour_accuracy.append(float(linereader[4][-1]))
                false_colour_accuracy.append(float(linereader[10][-1]))
        except FileNotFoundError:
            print(f"Warning: Could not find ResNet18 data for pruning step {step}")
    
    # Create the plot with the exact style from the provided image
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    
    # Plot tasks with colors matching the legend
    # From the legend image:
    # - silhouette is a light orange/salmon color
    # - colour is a burgundy/maroon color
    # - false_colour is a light cyan/turquoise color
    plt.plot(silhouette_accuracy, label='silhouette', linewidth=10, color='#ff9d5c', marker='s', markersize=6)  # Light orange for silhouette
    plt.plot(colour_accuracy, label='colour', linewidth=10, color='#CC0000', marker='s', markersize=6)  # Burgundy/maroon for colour
    plt.plot(false_colour_accuracy, label='false_colour', linewidth=10, color='#45c6d6', marker='s', markersize=6)  # Turquoise for false_colour
    
    # Set axis limits and labels
    plt.ylim(0, 1)
    plt.xlabel('Pruning step', fontsize=42)
    plt.ylabel('Accuracy', fontsize=42)
    
    # Add grid for better readability
    plt.grid(True, linestyle='-', alpha=0.1)
    
    # Adjust legend to match the provided example
    plt.legend(loc='lower left', fontsize=40, frameon=True)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_dir = os.path.join(base_dir, 'model_vs_human_performances', 'resnet18')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as PDF with high quality
    save_path = os.path.join(save_dir, 'resnet18_multiple_tasks.pdf')
    plt.savefig(save_path, 
               format='pdf', 
               bbox_inches='tight',
               dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    main()