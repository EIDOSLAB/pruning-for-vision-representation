from ast import arg
import random
import cv2
from doctest import OutputChecker
from pyclbr import Class
from re import A
from turtle import title

# from attr import attrib
import torch
import torch.nn.functional as F

from collections import OrderedDict
from glob import glob
from PIL import Image
from imagenet_classes import IMAGENET2012_CLASSES

import datetime
import os
import time
import warnings

import torch.nn.utils.prune as prune
import presets
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix

import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt

from torchvision import models
from torchvision import transforms

def compute_parameters(model):
        pruned_parameters = float(sum([torch.sum(module.weight == 0) for module in model.modules() if isinstance(module, nn.Conv2d)]))
        total_parameters = float(sum([module.weight.nelement() for module in model.modules() if isinstance(module, nn.Conv2d)]))
        return total_parameters - pruned_parameters


def main(args):
    
    parameters = []
    if 'train' in args.root_results_path:
        if not os.path.exists(f'{args.root_results_path}/parameters_{args.model}.txt'):
            glob_path = os.path.join(args.models_path, args.model.split('_')[0], 'model_epoch_89_pruning_iteration_*.pth') 
            print(glob_path)
            for i, checkpoint_path in enumerate(sorted(glob(glob_path))):
                print("Model filename: " + checkpoint_path)
                print(f"Resuming model from pruning iteration {i}")
                model = load_model(checkpoint_path, args, 1000)
                # print(model)
                model.eval()
                parameters.append(float(compute_parameters(model)))

            with open(f'{args.root_results_path}/parameters_{args.model}.txt', 'w') as f:
                for p in parameters:
                    f.write(str(p) + '\n')
                

            with open(f'{args.root_results_path}/parameters_{args.model}.txt', 'r') as f:
                parameters = f.readlines()

    with open(args.performance_path, 'r') as f:
        perfs = f.readlines()

    print(perfs)

    accs = []
    for perf in perfs[1:]:
        accs.append(float(perf.rstrip().split('\t')[0]))
    print(accs)

    with open(args.performance_path, 'r') as f:
        perfs = f.readlines()

    print(perfs)
    sparsities = []
    for perf in perfs[1:]:
        sparsities.append(float(perf.rstrip().split('\t')[1]))
    print(sparsities)

    corloc_performances_dilation_1 = []
    corloc_performances_dilation_2= []
    corloc_performances_dilation_4 = []
    if 'resnet50' in args.model:
        dilations = [2]
    else:
        dilations = [1]

    for pruning_iteration in range(args.model_max_prune+1):
        if 'resnet' in args.model:
            for dilation in dilations:
                result_file_path = os.path.join(args.root_results_path, f'LOST-{args.model}dilate{dilation}/results_iteration_{pruning_iteration:02}.txt')
                f = open(result_file_path, 'r')
                results = f.readline()
                f.close()
                if dilation == 1:
                    corloc_performances_dilation_1.append(float(results.rstrip().split(',')[1]))
                if dilation == 2:
                    corloc_performances_dilation_2.append(float(results.rstrip().split(',')[1]))
                if dilation == 4:
                    corloc_performances_dilation_4.append(float(results.rstrip().split(',')[1]))
        else:
            result_file_path = os.path.join(args.root_results_path, f'LOST-{args.model}/results_iteration_{pruning_iteration:02}.txt')
            f = open(result_file_path, 'r')
            results = f.readline()
            f.close()
            corloc_performances_dilation_1.append(float(results.rstrip().split(',')[1]))
    
    print(corloc_performances_dilation_1)
    print(corloc_performances_dilation_2)
    print(corloc_performances_dilation_4)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    color = 'tab:red'
    ax1.set_xlabel('Model Sparsity')
    ax1.set_ylabel('CorLoc performance %', color=color)
    if 'resnet50' in args.model:
        if not args.csv:
            ax1.plot(np.array(sparsities).astype('str'), corloc_performances_dilation_2, 'ro', color=color) # , label='Pruned Model with dilation 2', color=color)
            ax1.plot(np.array(sparsities).astype('str'), corloc_performances_dilation_2, color=color) 
        else:
            if 'train' in args.root_results_path:
                f = open(f'/scratch/lost/{args.model}_trainval_plot.csv', 'w')
            else:
                f = open(f'/scratch/lost/{args.model}_val_plot.csv', 'w')
            for i, j in zip(np.array(sparsities[:args.model_max_prune+1]).astype('str'), corloc_performances_dilation_2):
                f.write(str(i) + ',' + str(j) + '\n')
            f.close()
    else: 
        if not args.csv:
            ax1.plot(np.array(sparsities[:args.model_max_prune+1]).astype('str'), corloc_performances_dilation_1, 'ro', color=color) # , label='Pruned Model with dilation 2', color=color)
            ax1.plot(np.array(sparsities[:args.model_max_prune+1]).astype('str'), corloc_performances_dilation_1, color=color) 
        else:
            if 'train' in args.root_results_path:
                f = open(f'/scratch/lost/{args.model}_trainval_plot.csv', 'w')
            else:
                f = open(f'/scratch/lost/{args.model}_val_plot.csv', 'w')
            for i, j in zip(np.array(sparsities[:args.model_max_prune+1]).astype('str'), corloc_performances_dilation_1):
                f.write(str(i) + ',' + str(j) + '\n')
            f.close()
    
    ax1.set_xticklabels(labels = np.array(sparsities[:args.model_max_prune+1]).astype('str'), rotation=30)

    
    if not args.csv:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Acc1 for ImageNet validation', color=color)
        ax2.plot(np.array(sparsities).astype('str'), accs, 'bo')
        ax2.plot(np.array(sparsities).astype('str'), accs, color=color)
    else:

        f = open(f'/scratch/lost/{args.model}_acc1.csv', 'w')
        for i, j in zip(np.array(sparsities[:args.model_max_prune+1]).astype('str'), accs):
            f.write(str(i) + ',' + str(j) + '\n')
        f.close()

    if 'train' in args.root_results_path:
        if not args.csv:
            plt.title(f'{args.model} tested on trainval')
            plt.xticks(rotation=30)
            plt.tight_layout()
            ax1.legend()
            plt.savefig(f'/scratch/lost/{args.model}_trainval_plot.jpg')
            plt.close()
        
            plt.figure(figsize=(14, 6))
            plt.plot(np.array(sparsities[:args.model_max_prune+1]).astype('str'), parameters, 'o', label='Abs Parameters')
            plt.xlabel('Model Sparsity')


            plt.title(f'{args.model} Abs Parameters Number')
            plt.ylabel('Abs Parameters Number')
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.legend()
            plt.savefig(f'/scratch/lost/{args.model}_parameters.jpg')
            plt.close()
        else:
            f = open(f'/scratch/lost/{args.model}_parameters.csv', 'w')
            for i, j in zip(np.array(sparsities[:args.model_max_prune+1]).astype('str'), parameters):
                f.write(str(i) + ',' + str(j) + '\n')
            f.close()
    else:
        if not args.csv:
            plt.title(f'{args.model} tested on validation')
            plt.xticks(rotation=30)
            plt.tight_layout()
            ax1.legend()
            plt.savefig(f'/scratch/lost/{args.model}_val_plot.jpg')
            plt.close()


def load_model(checkpoint_path, args, num_classes=1000):
    if 'swin' in args.model:
        model = torchvision.models.swin_v2_t()
        pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
    else:

        model = torchvision.models.get_model(args.model.split('_')[0], weights=args.weights)
        if args.model == 'resnet50_imagenet':
            pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
        else:
            pruned_layer_types = [torch.nn.Conv2d]

    checkpoint = torch.load(checkpoint_path, map_location='cpu')["model"]

    correct_checkpoint = OrderedDict()
    for k in checkpoint:
        correct_checkpoint[k.replace("module.", "")] = checkpoint[k]

    try:
        model.load_state_dict(correct_checkpoint)
    except:
        for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
            prune.identity(module, "weight")

        try:
            model.load_state_dict(correct_checkpoint)

            for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
                prune.remove(module, "weight")
        except:
            raise RuntimeError()

    return model

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--performance-path", default=None, type=str, help="File with model performance in relation to its sparsity")
    parser.add_argument("--root-results-path", default=None, type=str, help="Root directory for results file, corloc test LOST")
    parser.add_argument("--model", default=None, type=str, help="Model Name")
    parser.add_argument("--models-path", default="/shared/datasets/classification/imagenet/", type=str, help="models path")
    parser.add_argument("--model-max-prune", default=None, type=int, help="Root directory for results file, corloc test LOST")
    parser.add_argument("--model-baseline", default=None, type=float, help="Performance for the chosen model in LOST corloc task")
    parser.add_argument("--dilation", default=2, type=int, help="Performance for the chosen model in LOST corloc task")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--csv", action="store_true", help="the weights enum name to load")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)