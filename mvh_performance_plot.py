import os 
from matplotlib import pyplot as plt
import csv
import matplotlib.colors as mcolors
import numpy as np

def main(args):
   # Set figure size and style
   plt.rcParams.update({
       'font.size': 14,
       'axes.labelsize': 42,
       'axes.titlesize': 42,
       'xtick.labelsize': 42,
       'ytick.labelsize': 42,
       'legend.fontsize': 25,
       'figure.figsize': (15, 10),  # Increased figure size
       'figure.dpi': 300
   })

   base_dir = "/scratch/tesi_magistrale/"
   
   # Load sparsity values from performance.txt
   sparsity_values = []
   sparsity_file = f"/scratch/tesi_magistrale/{args.model_name}/performance.txt"
   with open(sparsity_file, mode='r') as sparsity_f:
       # Skip header line
       next(sparsity_f)
       for line in sparsity_f:
           if line.strip():  # Skip empty lines
               parts = line.strip().split()
               if len(parts) >= 2:
                   sparsity = float(parts[1])
                   sparsity_values.append(sparsity)
   
   # Make sure we have enough sparsity values for all pruning steps
   if len(sparsity_values) < args.max_prune:
       print(f"Warning: Not enough sparsity values in {sparsity_file}. Found {len(sparsity_values)}, needed {args.max_prune}")
       # Fill remaining with placeholder values
       sparsity_values.extend([np.nan] * (args.max_prune - len(sparsity_values)))
   elif len(sparsity_values) > args.max_prune:
       # Truncate if we have too many
       sparsity_values = sparsity_values[:args.max_prune]

   edge_accuracy_top_1 = [] 
   silhouette_accuracy_top_1=[]
   cue_conflict_accuracy_top_1=[]
   colour_accuracy_top_1=[]
   contrast_accuracy_top_1=[]
   high_pass_accuracy_top_1=[]
   low_pass_accuracy_top_1=[]
   phase_scrambling_accuracy_top_1=[]
   power_equalisation_accuracy_top_1=[]
   false_colour_accuracy_top_1=[]
   rotation_accuracy_top_1=[]
   eidolonI_accuracy_top_1=[]
   eidolonII_accuracy_top_1=[]
   eidolonIII_accuracy_top_1=[]
   uniform_noise_accuracy_top_1=[]
   sketch_accuracy_top_1=[]
   stylized_accuracy_top_1=[]
   rn50_rotation = []


   for step in range(args.max_prune):
       print('Reading:')


       with open(base_dir + args.model_name + f'_pruning_step_{str(step)}.csv', mode='r') as csvfile:
           linereader = list(csv.reader(csvfile))
           print(linereader)
           edge_accuracy_top_1.append(float(linereader[1][-1]))
           silhouette_accuracy_top_1.append(float(linereader[2][-1]))
           cue_conflict_accuracy_top_1.append(float(linereader[3][-1]))
           colour_accuracy_top_1.append(float(linereader[4][-1]))
           contrast_accuracy_top_1.append(float(linereader[5][-1]))
           high_pass_accuracy_top_1.append(float(linereader[6][-1]))
           low_pass_accuracy_top_1.append(float(linereader[7][-1]))
           phase_scrambling_accuracy_top_1.append(float(linereader[8][-1]))
           power_equalisation_accuracy_top_1.append(float(linereader[9][-1]))
           false_colour_accuracy_top_1.append(float(linereader[10][-1]))
           rotation_accuracy_top_1.append(float(linereader[11][-1]))
           eidolonI_accuracy_top_1.append(float(linereader[12][-1]))
           eidolonII_accuracy_top_1.append(float(linereader[13][-1]))
           eidolonIII_accuracy_top_1.append(float(linereader[14][-1]))
           uniform_noise_accuracy_top_1.append(float(linereader[15][-1]))
           sketch_accuracy_top_1.append(float(linereader[16][-1]))
           stylized_accuracy_top_1.append(float(linereader[18][-1]))

   # Define 17 distinguishable colors for the different tests (all with solid line type)
   colors = [
       '#FF0000',  # Red
       '#0000FF',  # Blue
       '#00FF00',  # Green
       '#FF00FF',  # Magenta
       '#00FFFF',  # Cyan
       '#FFD700',  # Gold
       '#FF6600',  # Orange
       '#8B008B',  # Dark Magenta
       '#000000',  # Black
       '#006400',  # Dark Green
       '#8B0000',  # Dark Red
       '#4B0082',  # Indigo
       '#00008B',  # Dark Blue
       '#808000',  # Olive
       '#800080',  # Purple
       '#008080',  # Teal
       '#9932CC',  # Dark Orchid
   ]

   fig = plt.figure(figsize=(22, 10))  # Slightly wider to accommodate sparsity labels
   ax = fig.add_subplot(111)
   
   # Create a consistent mapping of test names to colors
   test_data = [
       (edge_accuracy_top_1, 'edge', colors[0]),
       (silhouette_accuracy_top_1, 'silhouette', colors[1]),
       (cue_conflict_accuracy_top_1, 'cue_conflict', colors[2]),
       (colour_accuracy_top_1, 'colour', colors[3]),
       (contrast_accuracy_top_1, 'contrast', colors[4]),
       (high_pass_accuracy_top_1, 'high_pass', colors[5]),
       (low_pass_accuracy_top_1, 'low_pass', colors[6]),
       (phase_scrambling_accuracy_top_1, 'phase_scrambling', colors[7]),
       (power_equalisation_accuracy_top_1, 'power_equalisation', colors[8]),
       (false_colour_accuracy_top_1, 'false_colour', colors[9]),
       (rotation_accuracy_top_1, 'rotation', colors[10]),
       (eidolonI_accuracy_top_1, 'eidolonI', colors[11]),
       (eidolonII_accuracy_top_1, 'eidolonII', colors[12]),
       (eidolonIII_accuracy_top_1, 'eidolonIII', colors[13]),
       (uniform_noise_accuracy_top_1, 'uniform_noise', colors[14]),
       (sketch_accuracy_top_1, 'sketch', colors[15]),
       (stylized_accuracy_top_1, 'stylized', colors[16]),
   ]
   
   # Plot each test with its assigned color and the same solid line style
   for data, label, color in test_data:
       plt.plot(data, label=label, color=color, linewidth=3, linestyle='solid')

   # Set x-ticks to show sparsity values with smaller font size
   plt.xticks(range(args.max_prune), [f"{val:.2f}" for val in sparsity_values], rotation=45, fontsize=34)
   
   # Set axis limits and labels
   plt.ylim(0, 1)
   plt.xlabel('Sparsity (%)', fontsize=42)
   plt.ylabel('Accuracy', fontsize=42)
   
   # Adjust legend
   plt.legend(bbox_to_anchor=(1.05, 1), 
             loc='upper left', 
             fontsize=40,
             frameon=True,
             ncol=5,  # Changed to 3 columns for better distribution
             edgecolor='black',
             fancybox=False,  # Sharp corners for better legibility
             labelspacing=0.7)  # More space between items for better readability
   
   # Add grid for better readability
   plt.grid(True, linestyle='--', alpha=0.3)
   
   # Add vertical lines at 0%, 50%, 90%, and 99% sparsity for reference (if they exist in our data)
   reference_sparsities = [0, 50, 90, 99]
   for ref_sparsity in reference_sparsities:
       # Find the closest sparsity value
       if sparsity_values:
           closest_idx = min(range(len(sparsity_values)), 
                            key=lambda i: abs(sparsity_values[i] - ref_sparsity))
           if abs(sparsity_values[closest_idx] - ref_sparsity) < 5:  # If we're within 5% of a reference point
               plt.axvline(x=closest_idx, color='gray', linestyle=':', alpha=0.5)
   
   # Adjust layout to prevent text cutoff
   plt.tight_layout()
   
   # Save as PDF with high quality
   save_path = os.path.join(base_dir, 'model_vs_human_performances', 
                           args.model_name, f'{args.model_name}_global.pdf')
   plt.savefig(save_path, 
               format='pdf', 
               bbox_inches='tight',
               dpi=300)
   plt.close()

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--model-name", default="resnet50", type=str, help="Chosen explainability method")
    parser.add_argument("--max-prune", default="26", type=int, help="Chosen explainability method")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)