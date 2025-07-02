import os

def main(args):
    images = ['2008_005309.jpg',  '2009_003338.jpg',  '2010_004855.jpg',  '2011_004631.jpg',  '2012_004256.jpg',
    '2008_005310.jpg',  '2009_003340.jpg',  '2010_004856.jpg',  '2011_004632.jpg',  '2012_004257.jpg',
    '2008_005313.jpg',  '2009_003343.jpg',  '2010_004857.jpg',  '2011_004635.jpg',  '2012_004258.jpg',
    '2008_005315.jpg',  '2009_003345.jpg',  '2010_004861.jpg',  '2011_004636.jpg',  '2012_004262.jpg',
    '2008_005316.jpg',  '2009_003346.jpg',  '2010_004865.jpg',  '2011_004638.jpg',  '2012_004267.jpg',
    '2008_005319.jpg',  '2009_003347.jpg',  '2010_004866.jpg',  '2011_004640.jpg',  '2012_004268.jpg',
    '2008_005321.jpg',  '2009_003348.jpg',  '2010_004868.jpg',  '2011_004645.jpg',  '2012_004270.jpg',
    '2008_005323.jpg',  '2009_003349.jpg',  '2010_004871.jpg',  '2011_004646.jpg',  '2012_004272.jpg','2009_003348.jpg']
    
    if 'resnet50' in args.model_name:
        dilation = 2
    else: 
        dilation = 1

    for pruning_iteration in range(1, args.model_max_prune+1):
        for img in images:
            cmd = f"python main_lost.py --arch {args.model_name} --output_dir ../../lost --models_dir /home/cassano/models --pruning_iteration {pruning_iteration} --patch_size 32 --image_path datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages/{img} --visualize pred --resnet_dilate {dilation}"
            os.system(cmd)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--performance-path", default=None, type=str, help="File with model performance in relation to its sparsity")
    parser.add_argument("--model-name", default=None, type=str, help="Model Name")
    parser.add_argument("--model-max-prune", default=None, type=int, help="Root directory for results file, corloc test LOST")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)