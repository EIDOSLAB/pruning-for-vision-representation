import datetime
import os
import time
import warnings
from collections import OrderedDict
from datasets import Dataset 

import torch.nn.utils.prune as prune
import presets
# import wandb
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
from glob import glob
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix


def evaluate(model, sparsity, criterion, data_loader, device, print_freq=100, log_suffix="", split='test', args=None, imgs= None, targets = None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    tot_acc1=0
    tot_acc5=0
    tot_loss=0   

    num_processed_samples = 0
    with torch.inference_mode():
        if data_loader is not None:
            for image, target in metric_logger.log_every(data_loader, print_freq, header):
                print(image)
                print(target)
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(image)
                loss = criterion(output, target)

                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                num_processed_samples += batch_size
        else:

            for image_id, image in enumerate(imgs):
                image = image.to(device, non_blocking=True)
                target = targets[image_id].to(device, non_blocking=True)
                output = model(image)
                loss = criterion(output, target)

                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                num_processed_samples += batch_size

        tot_acc1 += acc1.item()
        tot_acc5 += acc5.item()
        tot_loss += loss


#     if utils.is_main_process():
#         wandb.log({f"{split}/acc1" : tot_acc1/len(data_loader), "custom_x_axis":sparsity})
#         wandb.log({f"{split}/acc5":tot_acc5 / len(data_loader), "custom_x_axis":sparsity})
#         wandb.log({f"{split}/loss":tot_loss/len(data_loader), "custom_x_axis":sparsity})

    with open(f'{args.output_dir}/{args.model}_dilation_{args.resnet_dilate}.txt','a') as f:
        f.write(f"{split}/acc1  {tot_acc1/len(data_loader)}, custom_x_axis {sparsity}\n")
        f.write(f"{split}/acc5  {tot_acc5/len(data_loader)}, custom_x_axis {sparsity}\n")
        f.write(f"{split}/loss {tot_loss/len(data_loader)}, custom_x_axis {sparsity}\n")


    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


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

def compute_sparsity_convolutional(model):
        s = 100. * (float(sum([torch.sum(module.weight == 0) for module in model.modules() if isinstance(module, nn.Conv2d)])) / float(sum([module.weight.nelement() for module in model.modules() if isinstance(module, nn.Conv2d)])))
        return s

def compute_sparsity_linear(model):
        s = 100. * (float(sum([torch.sum(module.weight == 0) for module in model.modules() if isinstance(module, nn.Linear)])) / float(sum([module.weight.nelement() for module in model.modules() if isinstance(module, nn.Linear)])))
        return s

def compute_sparsity_global(model):
        s = 100. * (float(sum([torch.sum(module.weight == 0) for module in model.modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])) / float(sum([module.weight.nelement() for module in model.modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)])))
        return s

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

#     if utils.is_main_process():
#         run = wandb.init(
#             project=f"{args.model}-cassano_tesi",
#             name=f"{args.model}-pruning",
#             config={
#                 "architecture": f"{args.model}",
#                 "dataset" : "Imagenet-1K",
#                 "epochs" : args.epochs,
#             }
#         )

    # wandb.define_metric("custom_x_axis")
    # wandb.define_metric("*", step_metric="custom_x_axis")

    if 'VOC' not in args.data_path:
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
    else:
        dataset_test =  Dataset(args.dataset, args.set, args.no_hard)
        num_classes=1000

    glob_path = os.path.join(args.models_path, args.model, 'model_epoch_89_pruning_iteration_*.pth') 
    for i, checkpoint_path in enumerate(sorted(glob(glob_path))):

        print(checkpoint_path)
        print(f"Resuming model from pruning iteration {i}")
        model = load_model(checkpoint_path, args, num_classes)
        print(model)
        model.eval()
        model.to(device)
    
        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
        custom_keys_weight_decay = []
        if args.bias_weight_decay is not None:
            custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
        if args.transformer_embedding_decay is not None:
            for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
                custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

        parameters = utils.set_weight_decay(
            model,
            args.weight_decay,
            norm_weight_decay=args.norm_weight_decay,
            custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
        )
    
        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
    
        scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
            )
        elif args.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )
    
        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler
    
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
    
        model_ema = None
        if args.model_ema:
            # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
            alpha = 1.0 - args.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
    
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.test_only:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if scaler:
                scaler.load_state_dict(checkpoint["scaler"])
    
        if args.test_only:
            # We disable the cudnn benchmarking because it can noticeably affect the accuracy
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if model_ema:
                evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", args=args)
            else:
                evaluate(model, sparsity, criterion, data_loader_test, device=device, args=args)
            return
    
        sparsity = compute_sparsity_global(model)    
        print(sparsity)
#         if utils.is_main_process():
#             wandb.log({'sparsity':sparsity})
        if 'VOC' not in args.data_path:
            evaluate(model, sparsity, criterion, data_loader_test, device=device, args=args)
        else:
            # imgs = dataset_test
            # labels = dataset_test.extract_classes_VOC()
            # evaluate(model, sparsity, criterion=criterion,imgs=imgs, targets=labels data_loader=None, device=device, args=args)
            evaluate(model, sparsity, criterion, dataset_test.dataloader, device=device, args=args)
        print('End.')


def load_model(checkpoint_path, args, num_classes=1000):
    if 'swin' in args.model:
        model = torchvision.models.swin_v2_t()
        pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
    else:

        if args.resnet_dilate == 1:
            replace_stride_with_dilation = [False, False, False]
        elif args.resnet_dilate == 2:
            replace_stride_with_dilation = [False, False, True]
        elif args.resnet_dilate == 4:
            replace_stride_with_dilation = [False, True, True]
        model = torchvision.models.get_model(args.model, weights=args.weights, replace_stride_with_dilation=replace_stride_with_dilation, num_classes=num_classes) # , num_classes=num_classes, replace_stride_with_dilation=[False, False, False])
        if args.model == 'resnet50':
            pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
        else:
            pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]

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

    # Serve a togliere la fully connected per la classificazione
    model.eval()
    # model.to(device)

    return model

#def load_model(checkpoint_path, args, num_classes):
#    if args.model == 'swin':
#        model = torchvision.models.swin_v2_t()
#        pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
#    else:
#        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
#        if args.model == 'resnet50':
#            pruned_layer_types = [torch.nn.Conv2d, torch.nn.Linear]
#        else:
#            pruned_layer_types = [torch.nn.Conv2d]
#
#    checkpoint = torch.load(checkpoint_path)["model"]
#
#    correct_checkpoint = OrderedDict()
#    for k in checkpoint:
#        correct_checkpoint[k.replace("module.", "")] = checkpoint[k]
#
#    try:
#        model.load_state_dict(correct_checkpoint)
#    except:
#        for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
#            prune.identity(module, "weight")
#
#        try:
#            model.load_state_dict(correct_checkpoint)
#
#            for module in filter(lambda m: type(m) in pruned_layer_types, model.modules()):
#                prune.remove(module, "weight")
#        except:
#            raise RuntimeError()
#
#    return model


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="/shared/datasets/classification/imagenet/", type=str, help="dataset path")
    parser.add_argument("--models-path", default="/shared/datasets/classification/imagenet/", type=str, help="models path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
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
    parser.add_argument("--dataset", default=None, help="Use V2 transforms")
    parser.add_argument("--set", default=None, help="Use V2 transforms")
    parser.add_argument("--no-hard", action="store_true", help="Use V2 transforms")
    parser.add_argument("--resnet_dilate", type=int, default=2, help="Dilation level of the resnet model.")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)