import os 
import statistics
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import csv

def main(args):
    # Set larger font sizes globally with correct parameter names
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,    # Fixed from xticks to xtick
        'ytick.labelsize': 12,    # Fixed from yticks to ytick
        'legend.fontsize': 12
    })
    
    base_dir = "/scratch/"
    tests = {
        'edge accuracy (top-1)', 
        'silhouette accuracy (top-1)', 
        'cue-conflict accuracy (top-1)', 
        'colour accuracy (top-1)', 
        'contrast accuracy (top-1)', 
        'high-pass accuracy (top-1)', 
        'low-pass accuracy (top-1)', 
        'phase-scrambling accuracy (top-1)', 
        'power-equalisation accuracy (top-1)', 
        'false-colour accuracy (top-1)', 
        'rotation accuracy (top-1)', 
        'eidolonI accuracy (top-1)', 
        'eidolonII accuracy (top-1)', 
        'eidolonIII accuracy (top-1)', 
        'uniform-noise accuracy (top-1)', 
        'sketch accuracy (top-1)', 
        'sketch accuracy (top-5)', 
        'stylized accuracy (top-1)', 
        'stylized accuracy (top-5)', 
    }
    models_pruning = {
        'resnet18' : 27,
        'resnet50': 26,
        'swin' : 8,
        'vit_b_32' : 14,
    }
    
    for ii, test_name in enumerate(tests):
        plt.figure(figsize=(10, 8))
        
        for model in models_pruning.keys():
            values = []
            for step in range(models_pruning[model]):
                print(os.path.isfile(base_dir + model + f'_pruning_step_{str(step)}.csv'))
                with open(base_dir + model + f'_pruning_step_{str(step)}.csv', mode='r') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for ii, row in enumerate(csvreader):
                        if ii > 0:
                            print(row)
                            print(row[0])
                            if str(row[1] + ' ' + row[2]) == test_name:
                                print(row)
                                values.append(float(row[-1]))
            print(values)
            print(len(values))
            print(models_pruning[model])
            avg = statistics.mean(values)
            for _ in range(step, 28):
                values.append(avg)
            plt.plot(values, label=model, linewidth=2)
            
        plt.title(test_name, pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('Pruning Step', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(base_dir + test_name + '.png', bbox_inches='tight', dpi=300)
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