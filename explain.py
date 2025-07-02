from ast import arg
from atexit import register
from email.mime import image
import random
from tkinter import image_names
import cv2
from doctest import OutputChecker
from pyclbr import Class
from re import A
from turtle import title
from matplotlib.colors import Normalize

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
from torchvision.models import ViT_B_32_Weights, ViT_B_16_Weights

from captum.attr import IntegratedGradients
from captum.attr import LayerAttribution
from captum.attr import LRP
from captum.attr import LayerGradCam 
from captum.attr import NoiseTunnel
from captum.attr import GuidedGradCam 
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def main(args):
    if args.output_dir:
        print('Mk output dir')
        print(args.output_dir)
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    # Set the seed if provided and using untrained models
    if args.untrained and args.seed is not None:
        print(f"Setting seed to {args.seed} for model initialization")
        set_seed(args.seed)

    device = torch.device(args.device)
    print(device)
    
    val_dir = os.path.join(args.data_path, "val") 
    dataset_test, test_sampler = load_data_val_only(val_dir, args)
    
    num_classes = len(dataset_test.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    target_labels = list(IMAGENET2012_CLASSES.items())

    last_class = None
    class_counter = 0
    for batch_index, (image, target) in enumerate(data_loader_test):
        if class_counter == 201:
            break
        if target != last_class:
            class_counter += 1
            last_class = target
            batch_filenames, _ =  data_loader_test.dataset.samples[batch_index]
            if args.untrained:
                # Use the untrained vanilla model from PyTorch
                print(f"Using untrained vanilla model: {args.model} with seed: {args.seed}")
                model = load_untrained_model(args, num_classes)
                print(model)
                model.eval()
                # Create a directory name that includes the seed information
                dir_prefix = f"untrained_model_seed_{args.seed}" if args.seed is not None else "untrained_model"
                get_explainations(image, target, model, args, dir_prefix, target_labels, batch_index, batch_filenames, device=device)
            elif args.pruning_iteration == -1:
                glob_path = os.path.join(args.models_path, args.model, 'model_epoch_89_pruning_iteration_*.pth') 
                for i, checkpoint_path in enumerate(sorted(glob(glob_path))):
                    print("Model filename: " + checkpoint_path)
                    print(f"Resuming model from pruning iteration {i}")
                    model = load_model(checkpoint_path, args, num_classes)
                    print(model)
                    model.eval()
                    print(batch_filenames)
                    get_explainations(image, target, model, args, i, target_labels, batch_index, batch_filenames, device=device)
            elif args.pruning_method == 'snip':
                print(f"Resuming model pruned with snip")
                checkpoint_path = args.models_path
                print("Model filename: " + checkpoint_path)
                if 'vit' in args.model:
                    model = torchvision.models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
                    # model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                else:
                    model = load_model(checkpoint_path, args, num_classes)
                print(model)
                model.eval()
                get_explainations(image, target, model, args, args.pruning_iteration, target_labels, batch_index, batch_filenames)
            else:
                print(f"Resuming model from pruning iteration {args.pruning_iteration}")
                checkpoint_path = os.path.join(args.models_path, args.model, f'model_epoch_89_pruning_iteration_{args.pruning_iteration:02}.pth')
                print("Model filename: " + checkpoint_path)
                if 'vit' in args.model:
                    model = torchvision.models.vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
                    # model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
                else:
                    model = load_model(checkpoint_path, args, num_classes)
                print(model)
                model.eval()
                get_explainations(image, target, model, args, args.pruning_iteration, target_labels, batch_index, batch_filenames)
    
    print('End.')

def set_seed(seed):
    """
    Set all the random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_untrained_model(args, num_classes):
    """
    Load an untrained vanilla model from PyTorch's model zoo
    """
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    elif args.model == 'swin':
        model = torchvision.models.swin_v2_t(weights=None, num_classes=num_classes)
    elif 'vit' in args.model:
        if args.model == 'vit_b_16':
            model = torchvision.models.vit_b_16(weights=None, num_classes=num_classes)
        else:  # default to vit_b_32
            model = torchvision.models.vit_b_32(weights=None, num_classes=num_classes)
    else:
        model = torchvision.models.get_model(args.model, weights=None, num_classes=num_classes)
    
    # Move model to the specified device
    if args.device:
        model = model.to(args.device)
        
    return model

def get_explainations(image, target, model, args, pruning_iteration, target_labels, batch_index, batch_filenames, device='cuda'):
            # Move input to the same device as the model
            image = image.to(device)
            if isinstance(target, torch.Tensor):
                target = target.to(device)
                
            if args.model == 'swin':
                guided_gradCAM(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                gradCAM(model, [model.features[-1][-1].norm1], image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                # lrp(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                integrated_gradients(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
            elif 'vit' in args.model:
                vit_attention_heatmap(model, None, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames, device, patch_size=32)
                guided_gradCAM(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                gradCAM(model, [model.features[-1][-1].norm1], image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                # lrp(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                integrated_gradients(model, model.features[-1][-1].norm1, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
            else:
                guided_gradCAM(model, model.layer4, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                gradCAM(model, [model.layer4[-1]], image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                # lrp(model, model.layer4, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)
                integrated_gradients(model, model.layer4, image, target, args, pruning_iteration, target_labels, batch_index, batch_filenames)

def load_model(checkpoint_path, args, num_classes):
    if args.model == 'swin':
        model = torchvision.models.swin_v2_t()
        pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
    else:
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
        if args.model == 'resnet50':
            pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
        else:
            pruned_layer_types = [torch.nn.Conv2d, nn.Linear]

    checkpoint = torch.load(checkpoint_path)["model"]

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


def save_results(index, attribution, original_image, target, predicted, method_name, args, pruning_iteration, target_labels, batch_index, filename):

    # If pruning_iteration contains "untrained_model", use it as the directory name
    # This handles both the original "untrained" case and our new seeded case
    if isinstance(pruning_iteration, str) and "untrained_model" in pruning_iteration:
        dir_prefix = pruning_iteration
    else:
        dir_prefix = f'pruning_iteration_{pruning_iteration:02}'
    
    utils.mkdir(os.path.join(args.output_dir, dir_prefix, f'{method_name}'))
    dir = os.path.join(args.output_dir, dir_prefix, f'{method_name}', f'{target_labels[target][0]}')
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    print('Filename:')
    print(filename)
    filename = filename.split('/')[-1].split(".")[0]
    print('Filename:')
    print(filename)

    # For untrained models, we won't have performance metrics
    if not isinstance(pruning_iteration, str) or "untrained_model" not in pruning_iteration:
        # performance_file_path = os.path.join(args.output_dir, 'performance.txt')
        print('Recupero le performance')
        if args.pruning_method == "snip":
            performance_file_path = os.path.join("/scratch/snips/sparsity_914/", args.model, 'performance.txt')
        else:
            performance_file_path = os.path.join(args.models_path, args.model, 'performance.txt')
        with open(performance_file_path, 'r') as f:
            values = f.readlines()

        print(performance_file_path)
        print(values)

        sparsity = values[pruning_iteration+1].split("\t")[1]
        acc1 = values[pruning_iteration+1].split("\t")[0]
        print('Sparsity:' + str(sparsity))
        print('Acc1:' + str(acc1))
    else:
        sparsity = "N/A"
        acc1 = "N/A"

    # Denormalization
    mean = torch.tensor([0.4915, 0.4823, 0.4468])
    std = torch.tensor([0.2470, 0.2435, 0.2616])

    normalize = transforms.Normalize(mean.tolist(), std.tolist()) 

    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    np_original_image = unnormalize(original_image)
    np_original_image = np_original_image.squeeze().cpu().detach().numpy()

    # Extract seed information for title if available
    seed_info = ""
    if isinstance(pruning_iteration, str) and "seed_" in pruning_iteration:
        seed_str = pruning_iteration.split("seed_")[1]
        seed_info = f" - Seed: {seed_str}"

    if method_name == 'gradcam':

        print(original_image.shape)
        print(attribution.shape)
        
        fig = plt.figure(figsize=(11, 11))
        attr = plt.imshow(attribution,  alpha=0.95, cmap='jet')
        plt.colorbar(attr, location='bottom') # cmap=cv2.COLORMAP_JET)
        plt.imshow(np.transpose(np_original_image, (1, 2, 0)), alpha=.75)
        plt.title(f'Target Class: {target_labels[target][1]} \n - Predicted Class: {target_labels[np.argmax(predicted)][1]} \n - Model: {args.model} \n - {"Untrained" if "untrained_model" in str(pruning_iteration) else f"pruning iteration {pruning_iteration}"}{seed_info} - Method: {method_name} \n Sparsity: {sparsity} - Acc1: {acc1}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, dir_prefix, f'{method_name}/{target_labels[target][0]}/blended_heatmap_{filename}'))#  blended_heatmap_{index}_batch_{batch_index}.jpg'))
        plt.close()
        print('Salvata immagine originale GradCAM')
        
        fig = plt.figure(figsize=(11, 11))
        attr = plt.imshow(attribution, cmap='jet')
        np.save(os.path.join(args.output_dir, dir_prefix, f'{method_name}/{target_labels[target][0]}/{filename.split(".")[0]}_nparray.npy'), attribution)
        plt.colorbar(attr, location='bottom') # cmap=cv2.COLORMAP_JET)
        plt.title(f'Target Class: {target_labels[target][1]} \n - Predicted Class: {target_labels[np.argmax(predicted)][1]} \n - Model: {args.model} \n - {"Untrained" if "untrained_model" in str(pruning_iteration) else f"pruning iteration {pruning_iteration}"}{seed_info} - Method: {method_name} \n Sparsity: {sparsity} - Acc1: {acc1}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, dir_prefix, f'{method_name}/{target_labels[target][0]}/heatmap_{filename}'))
        plt.close()
        print('Salvata heatmap GradCAM')

    elif method_name == 'vit_attention':
        # Normalize image_metric to the range [0, 1]
        image_metric = attribution.detach().numpy()
        normalized_metric = Normalize(vmin=image_metric.min(), vmax=image_metric.max())(image_metric)
        
        # Apply the Reds colormap
        reds = plt.cm.Reds(normalized_metric)
        
        # Create the alpha channel
        alpha_max_value = 1.00  # Set your max alpha value
        
        # Adjust this value as needed to enhance lower values visibility
        gamma = 0.5  
        
        # Apply gamma transformation to enhance lower values
        enhanced_metric = np.power(normalized_metric, gamma)
        
        # Create the alpha channel with enhanced visibility for lower values
        alpha_channel = enhanced_metric * alpha_max_value
        
        # Add the alpha channel to the RGB data
        rgba_mask = np.zeros((image_metric.shape[0], image_metric.shape[1], 4))
        rgba_mask[..., :3] = reds[..., :3]  # RGB
        rgba_mask[..., 3] = alpha_channel  # Alpha
        
        # Convert the numpy array to PIL Image
        rgba_image = Image.fromarray((rgba_mask * 255).astype(np.uint8))
        
        # Save the image
        rgba_image.save(os.path.join(args.output_dir, dir_prefix, f'{method_name}/{target_labels[target][0]}/attention_heatmap_{filename}.png'))
        # rgba_image.save(os.path.join(args.'attention_mask.png')
        fig = plt.figure(figsize=(11, 11))
        plt.imshow(np.transpose(np_original_image, (1, 2, 0)), alpha=.75)
        attr = plt.imshow(rgba_image,  alpha=0.95, cmap='jet')
        plt.colorbar(attr, location='bottom') # cmap=cv2.COLORMAP_JET)
        # plt.imshow(rgba_mask, alpha=.90)
        plt.title(f'Target Class: {target_labels[target][1]} \n - Predicted Class: {target_labels[np.argmax(predicted)][1]} \n - Model: {args.model} \n - {"Untrained" if "untrained_model" in str(pruning_iteration) else f"pruning iteration {pruning_iteration}"}{seed_info} - Method: {method_name} \n Sparsity: {sparsity} - Acc1: {acc1}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, dir_prefix, f'{method_name}/{target_labels[target][0]}/blended_heatmap_{filename}.png'))#  blended_heatmap_{index}_batch_{batch_index}.jpg'))
        plt.close()

    else:
        attribution = attribution.squeeze().cpu().detach().numpy()
        print(attribution.shape)
        figure, axis = viz.visualize_image_attr(np.transpose(attribution, (1,2,0)),
                     np.transpose(np_original_image, (1,2,0)),
                     method='heat_map',
                     cmap='jet',
                     show_colorbar=False,  # Removed colorbar
                     sign='positive',
                     outlier_perc=1,
                     title=None  # Removed title
                     )
        # Remove ticks and tick labels
        axis.set_xticks([])
        axis.set_yticks([])
        # Optionally, also remove the axis borders
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        
        plt.tight_layout()
        print('File saved to:')
        print(os.path.join(args.output_dir, dir_prefix, f'{method_name}/{target_labels[target][0]}/heatmap_{filename}'))
        plt.savefig(os.path.join(args.output_dir, dir_prefix, f'{method_name}/{target_labels[target][0]}/heatmap_{filename}'))
        np.save(os.path.join(args.output_dir, dir_prefix, f'{method_name}/{target_labels[target][0]}/{filename.split(".")[0]}_nparray.npy'), attribution)


def vit_attention_heatmap(model, layer, image_batch, target_batch, args, pruning_iteration, target_labels, batch_index, batch_filenames, device, patch_size=16, register_tokens=0):
    # Ensure image_batch is on the same device as the model
    device = next(model.parameters()).device
    image_batch = image_batch.to(device)

    pred = model(image_batch)
    pred = pred.detach().cpu().numpy()
    predicted_class_numbers = []
    for p in pred:
        predicted_class_numbers.append(np.argmax(p))

    predicted_class_numbers = torch.tensor(predicted_class_numbers, device=device)

    for i, img in enumerate(image_batch):
        print(img.shape)

        original_w, original_h = img.shape[-2], img.shape[-1]

        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h]

        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        img = img.unsqueeze(0)
        img = img.to(device)
        attention = torch.load('/home/cassano/attention_vit.pt')
        print('Shape attenzione')
        print(attention)
        # attention = attention.unsqueeze(0)
        print(attention.shape)
        # exit()
        number_of_heads = attention.shape[1]
        attention = attention[0, :, 0, 1 +register_tokens:].reshape(number_of_heads, -1)

        attention = attention.reshape(number_of_heads, w_featmap, h_featmap)

        # attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode = "nearest")[0].cpu() # originale
        attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode = "bilinear")[0].cpu() 
        attention = torch.sum(attention, dim=0)
        attention_of_image = nn.functional.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(original_h, original_w), mode='bilinear', align_corners=False)
        attention_of_image = attention_of_image.squeeze()

        save_results(i, attention_of_image, image_batch[i], target_batch[i], pred[i], 'vit_attention', args, pruning_iteration, target_labels, batch_index, batch_filenames)


def guided_gradCAM(model, layer, image_batch, target_batch, args, pruning_iteration, target_labels, batch_index, batch_filenames):
    guided_gc = GuidedGradCam(model, layer)

    # Ensure image_batch is on the same device as the model
    device = next(model.parameters()).device
    image_batch = image_batch.to(device)
    
    pred = model(image_batch)
    print(image_batch.shape)
    pred = pred.detach().cpu().numpy()
    predicted_class_numbers = []
    for p in pred:
        predicted_class_numbers.append(np.argmax(p))

    predicted_class_numbers = torch.tensor(predicted_class_numbers, device=device)
    attribution = guided_gc.attribute(image_batch, predicted_class_numbers)

    print(attribution.shape)
    for i in range(len(image_batch)):
        # if i == 6:
        #     break
        try:
#             print('Predicted: ' + str(np.argmax(pred[i])))
#             print('Target: ' + str(target_batch[i]))
            # save_results(i, attribution[i], image_batch[i], target_batch[i], pred[i], 'guided_gradcam', args, pruning_iteration, target_labels, batch_index, batch_filenames)
            save_results(i, attribution[i], image_batch[i], target_batch[i], pred[i], 'guided_gradcam', args, pruning_iteration, target_labels, batch_index, batch_filenames)
        except Exception as msg:
            print(msg)

def reshape_transform(tensor, height=7, width=7):
    print(tensor.shape)
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(3))
    result = result.transpose(2, 3).transpose(1, 2)
    print(tensor.shape)
    return result

def gradCAM(model, layer, image_batch, target_batch, args, pruning_iteration, target_labels, batch_index, batch_filenames):
    if args.model == 'swin':
        cam = GradCAM(model=model, target_layers=layer, reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=layer)

    # Ensure image_batch is on the same device as the model
    device = next(model.parameters()).device
    image_batch = image_batch.to(device)
    
    pred = model(image_batch)
    pred = pred.detach().cpu().numpy()

    # Required By GradCAM
    new_targets = []
    for p in pred:
        new_targets.append(ClassifierOutputTarget(np.argmax(p)))

    attribution = cam(input_tensor=image_batch, targets=new_targets) #  , eigen_smooth=True, aug_smooth=True)
    for i in range(len(image_batch)):
        if i == 6:
            break
        try:
#             print('Predicted: ' + str(np.argmax(pred[i])))
#             print('Target: ' + str(target_batch[i]))
            print('Entro in salvataggio risultati GradCAM')
            save_results(i, attribution[i], image_batch[i], target_batch[i], pred[i], 'gradcam', args, pruning_iteration, target_labels, batch_index, batch_filenames)
        except Exception as msg:
            print(msg)

def lrp(model, layer, image_batch, target_batch, args, pruning_iteration, target_labels, batch_index, batch_filenames):
    # layers = list(model.named_modules())
    # num_layers = len(layers)

    # for idx_layer in range(1, num_layers):
    #     if idx_layer <= 16:
    #         setattr(layers[idx_layer], "rule", GammaRule())
    #     elif 17 <= idx_layer <= 30:
    #         setattr(layers[idx_layer], "rule", EpsilonRule())
    #     elif idx_layer >= 31:
    #         setattr(layers[idx_layer], "rule", EpsilonRule(epsilon=0))
    
    lrp = LRP(model)

    # Ensure image_batch is on the same device as the model
    device = next(model.parameters()).device
    image_batch = image_batch.to(device)
    
    pred = model(image_batch)
    pred = pred.detach().cpu().numpy()

    predicted_class_numbers = []
    for p in pred:
        predicted_class_numbers.append(np.argmax(p))

    predicted_class_numbers = torch.tensor(predicted_class_numbers, device=device)
    attribution = lrp.attribute(image_batch, predicted_class_numbers)
    for i in range(len(image_batch)):
        if i == 6:
            break
        try:
#             print('Predicted: ' + str(np.argmax(pred[i])))
#             print('Target: ' + str(target_batch[i]))
            save_results(i, attribution[i], image_batch[i], target_batch[i], pred[i], 'lrp', args, pruning_iteration, target_labels, batch_index, batch_filenames)
        except Exception as msg:
            print(msg)

def integrated_gradients(model, layer, image_batch, target_batch, args, pruning_iteration, target_labels, batch_index, batch_filenames):
    ig = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(ig)

    # Ensure image_batch is on the same device as the model
    device = next(model.parameters()).device
    image_batch = image_batch.to(device)
    
    pred = model(image_batch)
    pred = pred.detach().cpu().numpy()

    predicted_class_numbers = []
    for p in pred:
        predicted_class_numbers.append(np.argmax(p))

    predicted_class_numbers = torch.tensor(predicted_class_numbers, device=device)
    attribution = noise_tunnel.attribute(nt_samples_batch_size = 1, inputs=image_batch, target=predicted_class_numbers, nt_samples=2, nt_type='smoothgrad')

    for i in range(len(image_batch)):
        if i == 6:
            break
        try:
#             print('Predicted: ' + str(np.argmax(pred[i])))
#             print('Target: ' + str(target_batch[i]))
            save_results(i, attribution[i], image_batch[i], target_batch[i], pred[i], 'integrated_gradients', args, pruning_iteration, target_labels, batch_index, batch_filenames)
        except Exception as msg:
            print(msg)


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data_val_only(valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_test, test_sampler

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--untrained", action="store_true", help="Use an untrained vanilla model")
    parser.add_argument("--seed", default=None, type=int, help="Random seed for reproducibility")
    parser.add_argument("--expl-method", default="ggc", type=str, help="Chosen explainability method")
    parser.add_argument("--data-path", default="/shared/datasets/classification/imagenet/", type=str, help="dataset path")
    parser.add_argument("--models-path", default="/shared/datasets/classification/imagenet/", type=str, help="models path")
    parser.add_argument("--pruning-iteration", default=0, type=int, help="Pruning Iteration")
    parser.add_argument("--pruning-method", default='lrr', type=str, help="Pruning Method")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="/scratch", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)