# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random
import pickle
import math
import xml.etree.ElementTree as ET
from imagenet_classes import IMAGENET2012_CLASSES    
import vision_transformer

# from vit_pytorch import ViT

from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.prune as prune
import torchvision
from dino.vision_transformer import vit_base

from tqdm import tqdm
from PIL import Image
# from torchvision.models.vision_transformer import interpolate_embeddings
from vision_transformer import interpolate_embeddings

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_fms, visualize_predictions, visualize_seed_expansion
from object_discovery import lost, detect_box, dino_seg

class ResNetBottom(nn.Module):
    # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        print('Feature Map shape:')
        print(x.shape)
        return x

def interpolate_vit(base_vit, device, img_shape, args, first=False):

    model = base_vit
    # Rendo il pruning permanente, necessario per le key
    if args.pruning_iteration > 0 and not first:
        pruned_layer_types =  [torch.nn.Conv2d, torch.nn.Linear]
        for module in model.modules():
            # isinstance permette di gestiore anche NonDynamicallyQuantizableLinear che è una sottoclasse di Linear, che con type non veniva maskerata
            for t in pruned_layer_types:
                if isinstance(module, t) and prune.is_pruned(module): 
                    prune.remove(module, "weight")
                module.to(device)


    print('Shape passata')
    print(img_shape)

    # patch_size = 32
    patch_size = 16
    # model_interpolated = torchvision.models.vit_b_32(pretrained=False, image_size=img_shape)
    # model_interpolated = vision_transformer.vit_b_32(pretrained=False, image_size=img_shape)
    # model_interpolated = vision_transformer.vit_b_16(pretrained=True, image_size=img_shape)
    model_interpolated = vision_transformer.vit_b_16(pretrained=False, image_size=img_shape)


    state_dict_new = interpolate_embeddings(img_shape, patch_size, model.state_dict())
    model_interpolated.load_state_dict(state_dict_new)

    if args.pruning_iteration > 0 and not first:
        pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
        for module in model_interpolated.modules():
            # isinstance permette di gestiore anche NonDynamicallyQuantizableLinear che è una sottoclasse di Linear, che con type non veniva maskerata
            for t in pruned_layer_types:
                if isinstance(module, t): 
                    prune.identity(module, "weight")
                module.to(device)

    model = model_interpolated

    model.eval()
    model.to(device)

    return model

def load_model(checkpoint_path, args, device, num_classes=1000, base_vit_interpolated = None):

    if args.model == 'swin':
        model = torchvision.models.swin_v2_t()
        pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
    else:
        if 'vit' in args.model:
            if base_vit_interpolated is not None:
                model = base_vit_interpolated
            else:
                # model = torchvision.models.vit_b_32(image_size=224)
                model = vision_transformer.vit_b_32(image_size=224)
        else:
            model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)

        if args.model == 'resnet50':
            pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
        elif 'vit' in args.model:
            pruned_layer_types = [torch.nn.Linear, torch.nn.Conv2d] # , torch.nn.Linear.NonDynamicallyQuantizableLinear]
        else:
            pruned_layer_types = [torch.nn.Conv2d]

    print('Pruned Layers Type')
    print(pruned_layer_types)
    for child in model.named_children():
        print(child)


    checkpoint = torch.load(checkpoint_path, map_location=device)["model"]

    correct_checkpoint = OrderedDict()
    if 'vit' in args.model:
        for k in checkpoint:
            correct_checkpoint[k.replace("module.", "")] = checkpoint[k]

        try:
            model.load_state_dict(correct_checkpoint)
        except:
            # for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
            for module in model.modules():
                # isinstance permette di gestiore anche NonDynamicallyQuantizableLinear che è una sottoclasse di Linear, che con type non veniva maskerata
                for t in pruned_layer_types:
                    if isinstance(module, t): 
                        prune.identity(module, "weight")
                    module.to(device)
            
    
            try:
                model.load_state_dict(correct_checkpoint)
    
                for module in model.modules():
                    # isinstance permette di gestiore anche NonDynamicallyQuantizableLinear che è una sottoclasse di Linear, che con type non veniva maskerata
                    for t in pruned_layer_types:
                        if isinstance(module, t): 
                            prune.remove(module, "weight")
                        module.to(device)
#                 for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
#                     prune.remove(module, "weight")
#                     module.to(device)
            except:
                raise RuntimeError()
    else:
        for k in checkpoint:
            correct_checkpoint[k.replace("module.", "")] = checkpoint[k]
            # print(correct_checkpoint)
            # exit()
    
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

    # return model

    # Serve a togliere la fully connected per la classificazione
    if "resnet" in model_name:
        model = ResNetBottom(model)
    # Da fare anche per gli altri modelli

    # Per cuda out of memory
#    for param in model.features.parameters():
#        param.requires_grad = False

#     new_dict = interpolate_embeddings(image_size=(512, 512), patch_size=32, model_state=model.state_dict())
#     model = torchvision.models.vit_b_32(image_size=512)
#     model.load_state_dict(new_dict)

    model.eval()
    model.to(device)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unsupervised object discovery with LOST.")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "resnet50",
            "resnet18_imagenet",
            "swin_imagenet",
            "vgg16_imagenet",
            "resnet50_imagenet",
            "vit_b_32_imagenet",
            "vit_b_16_imagenet",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["fms", "seed_expansion", "pred", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # For ResNet dilation
    parser.add_argument("--resnet_dilate", type=int, default=2, help="Dilation level of the resnet model.")

    # LOST parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)
    parser.add_argument("--pruning_iteration", type=int, default=0)
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--models_dir", default='/home/cassano/models', type=str, help="models path")

    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    if args.resnet_dilate > 1:
        torch.backends.cudnn.deterministic=False
        torch.backends.cudnn.benchmark = True

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # perchè è piena la gpu di picasso
    # device = 'cpu'
    print(device)
    
    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with LOST
        exp_name = f"LOST-{args.arch}"
        if "resnet" in args.arch:
            exp_name += f"dilate{args.resnet_dilate}"
        elif "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"

    print(f"Running LOST on the dataset {dataset.name} (exp: {exp_name})")


    if 'vit' not in args.arch:
        model_name = args.arch.split('_')[0]
        checkpoint_path = os.path.join(args.models_dir, model_name, f"model_epoch_89_pruning_iteration_{args.pruning_iteration:02}.pth") # '/home/cassano/models/{model_name}/model_epoch_89_pruning_iteration_{args.pruning_iteration:02}.pth'
        print(checkpoint_path)
        args.model = model_name
        print('Model:')
        print(model_name)
        model = load_model(checkpoint_path=checkpoint_path, args=args, num_classes=1000,  device=device)

    else:
        model_name = args.arch.replace('_imagenet', '')
        checkpoint_path = os.path.join(args.models_dir, model_name, f"{model_name}_epoch_299_pruning_iteration_{args.pruning_iteration:02}.pth") # '/home/cassano/models/{model_name}/model_epoch_89_pruning_iteration_{args.pruning_iteration:02}.pth'
        print(checkpoint_path)
        args.model = model_name
        # model = load_model(checkpoint_path, args, model_name, device=device)
        print('Eseguo')

        # base_vit = torchvision.models.vit_b_32(pretrained=True, image_size=224)
        # base_vit = vision_transformer.vit_b_32(pretrained=True, image_size=224)
        base_vit = vision_transformer.vit_b_16(pretrained=True, image_size=224)
        model = interpolate_vit(base_vit=base_vit, device=device, img_shape=(224, 224), args=args, first = True)
        # model = load_model(checkpoint_path, args, model_name, device=device, base_vit_interpolated=model)
    
    # Per limitare la memoria usata nella GPU
    # for param in model.features.parameters():
    #     param.requires_grad = False

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/visualizations/{model_name}/pruning_iteration_{args.pruning_iteration:02}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))
    
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        init_image_size = img.shape

        print('Input shape')
        print(img.shape)

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])

        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        if device != 'cpu':
            img = img.cuda(non_blocking=True)

        if 'vit' in args.model:
            # model = load_model(checkpoint_path, args, device=device, base_vit_interpolated=None)
            model = torchvision.models.vit_b_16(pretrained=True, image_size=224)
            model = interpolate_vit(base_vit=model, device=device, img_shape=(img.shape[1], img.shape[2]), args=args)

        if device != 'cpu':
            img = img.cuda(non_blocking=True)

        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        print((w_featmap, h_featmap))

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue


        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if "vit" in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                # model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                # attentions = model.get_last_selfattention(img[None, :, :, :])
                pred = model(img[None, :, :, :])
                # img = img.to(device)
                target_labels = list(IMAGENET2012_CLASSES.items())
                print(target_labels[np.argmax(pred.cpu().numpy())][1])

                attentions = torch.load('/scratch/attention_vit.pt')
                feat_out_qkv = torch.load('/scratch/qkv_vit.pt')
                print("AT shape")
                print(attentions.shape)
                print("feat out shape")
                # print(feat_out["qkv"].shape)
                print(feat_out_qkv.shape)
                # Scaling factor
                scales = [args.patch_size, args.patch_size]
                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # w_featmap = int(math.sqrt(attentions.shape[-2]-1)) 
                # h_featmap = int(math.sqrt(attentions.shape[-2]-1)) 

                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        # feat_out["qkv"]
                        feat_out_qkv
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    print("feat out shape")
                    print(k.shape)

                    # Modality selection
                    if args.which_features == "k":
                        feats = k[:, 1:, :]
                    elif args.which_features == "q":
                        feats = q[:, 1:, :]
                    elif args.which_features == "v":
                        feats = v[:, 1:, :]

            elif "swin" in args.arch:
                # Store the outputs of qkv layer from the last attention layer
                feat_out = {}
                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output
                # model._modules["features.7.1.attn.qkv"].register_forward_hook(hook_fn_forward_qkv)
                model._modules["features"][7][1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                _ = model(img[None, :, :, :])

                attentions = torch.load('/home/cassano/last_attention_swin.pt')
                feat_out_qkv = torch.load('/home/cassano/qkv_swin.pt')
                print("AT shape")
                print(attentions.shape)
                print("feat out shape")
                print(feat_out_qkv.shape)

                w_featmap = int(math.sqrt(attentions.shape[-2])) # 8
                h_featmap = int(math.sqrt(attentions.shape[-1])) # 8
                print(w_featmap)

                # Scaling factor
                # scales = [args.patch_size, args.patch_size]


                scales = [
                    float(img.shape[1]) / args.patch_size,
                    float(img.shape[2]) / args.patch_size,
                ]

                # Originale 
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2] # Number of tokens


                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    pred = np.asarray(pred)
                else:
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        # feat_out["qkv"]
                        feat_out_qkv
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    print("feat out shape")
                    print(k.shape)


                    # Modality selection - a differenza di ViT, qui non c'è il token extra CLS, quindi parto da 0
                    if args.which_features == "k":
                        feats = k[:, 0:, :]
                    elif args.which_features == "q":
                        feats = q[:, 0:, :]
                    elif args.which_features == "v":
                        feats = v[:, 0:, :]
                    
                    print(feats.shape)

            elif "resnet" in args.arch:
                x = model.forward(img[None, :, :, :])
                d, w_featmap, h_featmap = x.shape[1:]
                print(x.shape)
                feats = x.reshape((1, d, -1)).transpose(2, 1)
                print(feats.shape)
                # Apply layernorm
                print(feats)
                print(torch.max(feats))
                layernorm = nn.LayerNorm(feats.size()[1:]).to(device)
                feats = layernorm(feats)
                print(feats)
                print(torch.max(feats))
                # exit()
                # Scaling factor
                scales = [
                    float(img.shape[1]) / x.shape[2],
                    float(img.shape[2]) / x.shape[3],
                ]
            elif "vgg16" in args.arch:
                x = model.forward(img[None, :, :, :])
                d, w_featmap, h_featmap = x.shape[1:]
                feats = x.reshape((1, d, -1)).transpose(2, 1)
                # Apply layernorm
                layernorm = nn.LayerNorm(feats.size()[1:]).to(device)
                feats = layernorm(feats)
                # Scaling factor
                scales = [
                    float(img.shape[1]) / x.shape[2],
                    float(img.shape[2]) / x.shape[3],
                ]
            else:
                raise ValueError("Unknown model.")

        # ------------ Apply LOST -------------------------------------------
        if not args.dinoseg:
            pred, A, scores, seed = lost(
                feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                k_patches=args.k_patches,
            )

            # ------------ Visualizations -------------------------------------------
            if args.visualize == "fms":
                visualize_fms(A.clone().cpu().numpy(), seed, scores, [w_featmap, h_featmap], scales, vis_folder, im_name)

            elif args.visualize == "seed_expansion":
                image = dataset.load_image(im_name)

                # Before expansion
                pred_seed, _ = detect_box(
                    A[seed, :],
                    seed,
                    [w_featmap, h_featmap],
                    scales=scales,
                    initial_im_size=init_image_size[1:],
                )
                visualize_seed_expansion(image, pred, seed, pred_seed, scales, [w_featmap, h_featmap], vis_folder, im_name)

            elif args.visualize == "pred":
                annotation = inp[1].replace("JPEGImages", "Annotations").replace("jpg", "xml")
                annotation = os.path.join(annotation)
                print(annotation)
                tree = ET.parse(annotation)
                root = tree.getroot()
                print(root)
                # print(root[2].getchildren())
                print(root[2].iter())
                # root_children = [str(child) for child in root[2].getchildren()]
                root_children = [str(child) for child in root[2].iter()]
                for id, name in enumerate(root_children):
                    if 'bndbox' in name:
                        break

                print(root[2][id][0].text)
                x1y1x2y2 = [
                    int(root[2][id][1].text),
                    int(root[2][id][3].text),
                    int(root[2][id][0].text),
                    int(root[2][id][2].text),
#                     int(obj["xmin"]),
#                     int(obj["ymin"]),
#                     int(obj["xmax"]),
#                     int(obj["ymax"]),
                ]
                # Original annotations are integers in the range [1, W or H]
                # Assuming they mean 1-based pixel indices (inclusive),
                # a box with annotation (xmin=1, xmax=W) covers the whole image.
                # In coordinate space this is represented by (xmin=0, xmax=W)
                # x1y1x2y2[0] -= 1
                # x1y1x2y2[1] -= 1

                image = dataset.load_image(im_name)
                visualize_predictions(image, pred, seed, scales, [w_featmap, h_featmap], vis_folder, im_name, gt_bbxs = x1y1x2y2)

        # Save the prediction
        preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))

        if torch.any(ious >= 0.5):
            corloc[im_id] = 1

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")
    
        # break


    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, f'results_iteration_{args.pruning_iteration:02}.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)

