from email.mime import image
from matplotlib import pyplot as plt
import matplotlib.animation as animation 
from PIL import Image
import matplotlib.image as mpimg
from glob import glob
import os
import math 

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if file.endswith('.png'):
                return file

def main(args):
    n_rows = 5
    n_cols = 6
    
    fig_h, ax_h = plt.subplots(n_rows, n_cols, figsize=(40, 20))
    fig_b, ax_b = plt.subplots(n_rows, n_cols, figsize=(40, 20))
    
    classes = ['n01440764', 'n01531178', 'n01614925', 'n01664065', 'n01689811', 'n01729977', 'n01753488', 'n01775062', 'n01818515', 'n01855672', 'n01924916', 'n01981276', 'n02009912', 'n02037110', 'n02086079', 'n02089078', 'n02091831', 'n02094433', 'n02097209', 
               'n01443537', 'n01532829', 'n01616318', 'n01665541', 'n01692333', 'n01734418', 'n01755581', 'n01776313', 'n01819313', 'n01860187', 'n01930112', 'n01983481', 'n02011460', 'n02051845', 'n02086240', 'n02089867', 'n02092002', 'n02095314', 'n02097298', 'n01484850', 'n01534433', 'n01622779', 'n01667114', 
               'n01693334', 'n01735189', 'n01756291', 'n01784675', 'n01820546', 'n01871265', 'n01943899', 'n01984695', 'n02012849', 'n02056570', 'n02086646', 'n02089973', 'n02092339', 'n02095570', 'n01491361', 
               'n01537544', 'n01629819', 'n01667778', 'n01694178', 'n01737021', 'n01768244', 'n01795545', 'n01824575', 'n01872401', 'n01944390', 'n01985128', 'n02013706', 'n02058221', 'n02086910', 'n02090379', 'n02093256', 'n02095889', 'n01494475', 'n01558993', 'n01630670', 'n01669191', 'n01695060', 'n01739381', 'n01770081', 'n01796340', 'n01828970', 'n01873310', 'n01945685', 'n01986214', 'n02017213', 'n02066245', 'n02087046', 'n02090622', 'n02093428', 'n02096051', 'n01496331', 'n01560419', 'n01631663', 'n01675722', 'n01697457', 'n01740131', 'n01770393', 'n01797886', 'n01829413', 'n01877812', 'n01950731', 'n01990800', 'n02018207', 'n02071294', 'n02087394', 'n02090721', 'n02093647', 'n02096177', 'n01498041', 'n01580077', 'n01632458', 'n01677366', 'n01698640', 'n01742172', 'n01773157', 'n01798484', 'n01833805', 'n01882714', 'n01955084', 'n02002556', 'n02018795', 'n02074367', 'n02088094', 'n02091032', 'n02093754', 'n02096294', 'n01514668', 'n01582220', 'n01632777', 'n01682714', 'n01704323', 'n01744401', 'n01773549', 'n01806143', 'n01843065', 'n01883070', 'n01968897', 'n02002724', 'n02025239', 'n02077923', 'n02088238', 'n02091134', 'n02093859', 'n02096437', 'n01514859', 'n01592084', 'n01641577', 'n01685808', 'n01728572', 'n01748264', 'n01773797', 'n01806567', 'n01843383', 'n01910747', 'n01978287',
                 'n02006656', 'n02027492', 'n02085620', 'n02088364', 'n02091244', 'n02093991', 'n02096585', 'n01518878', 'n01601694', 'n01644373', 'n01687978', 'n01728920', 'n01749939', 'n01774384', 'n01807496', 'n01847000', 'n01914609', 'n01978455', 'n02007558', 'n02028035', 'n02085782', 'n02088466', 'n02091467', 'n02094114', 'n02097047', 'n01530575', 'n01608432', 'n01644900', 
                 'n01688243', 'n01729322', 'n01751748', 'n01774750', 'n01817953', 'n01855032', 'n01917289', 'n01980166', 'n02009229', 'n02033041', 'n02085936', 'n02088632', 'n02091635', 'n02094258', 'n02097130']
    glob_path = os.path.join(args.expl_img_path, f'{args.model}', 'pruning_iteration_*') 
    files_heatmap_gif = []
    files_blended_heatmap_gif = []
    if args.batch_index is not None:
        for idx, path in enumerate(sorted(glob(glob_path))): 
            print(path)
    
            heatmap_name = files(path)
            heatmap_complete_path = path + f'/{args.expl_method}/{heatmap_name}'
            print(heatmap_complete_path)
            exit()
            files_heatmap_gif.append(heatmap_complete_path)
            ax_h[math.floor(idx/n_cols), (idx-(math.floor(idx/n_cols)*n_cols))].imshow(mpimg.imread(heatmap_complete_path))
    
            if args.expl_method == 'gradcam':
                blended_heatmap_complete_path = path + f'/{args.expl_method}/blended_heatmap_{args.image_index}_batch_{args.batch_index}.jpg'
                files_blended_heatmap_gif.append(blended_heatmap_complete_path)
                ax_b[math.floor(idx/n_cols), (idx-(math.floor(idx/n_cols)*n_cols))].imshow(mpimg.imread(blended_heatmap_complete_path))
    
        fig_h.suptitle(args.expl_method)
        fig_h.savefig(os.path.join(args.output_dir, 'visualizations', f'{args.model}', f'heatmap_global_{args.model}_{args.expl_method}_{args.image_index}_batch_{args.batch_index}.pdf'))
    
        if args.expl_method == 'gradcam':
            fig_b.suptitle(args.expl_method)
            fig_b.savefig(os.path.join(args.output_dir, 'visualizations', f'{args.model}', f'blended_heatmap_global_{args.model}_{args.expl_method}_{args.image_index}_batch_{args.batch_index}.pdf'))
    
        frames = [Image.open(image) for image in files_heatmap_gif]
        frame_one = frames[0]
        frame_one.save(os.path.join(args.output_dir, 'visualizations', f'{args.model}', f'heatmap_global_{args.model}_{args.expl_method}_{args.image_index}_batch_{args.batch_index}.gif'), format="GIF", append_images=frames, save_all=True, duration=550, loop=0)
    
        if args.expl_method == 'gradcam':
            frames = [Image.open(image) for image in files_blended_heatmap_gif]
            frame_one = frames[0]
            frame_one.save(os.path.join(args.output_dir, 'visualizations', f'{args.model}', f'blended_heatmap_global_{args.model}_{args.expl_method}_{args.image_index}_batch_{args.batch_index}.gif'), format="GIF", append_images=frames, save_all=True, duration=550, loop=0)

    else:
        model = args.model
        for expl_method in ['gradcam', 'lrp', 'guided_gradcam', 'integrated_gradients']:
            for cls in classes:
                n_rows = 5
                n_cols = 6

                fig_h, ax_h = plt.subplots(n_rows, n_cols, figsize=(40, 20))
                fig_b, ax_b = plt.subplots(n_rows, n_cols, figsize=(40, 20))

                files_heatmap_gif = []
                files_blended_heatmap_gif = []

                for idx, path in enumerate(sorted(glob(glob_path))): # Spans on all the pruning iterations.
                    path = os.path.join(path, expl_method, cls)
                    
                    print(path)
                
                    try:
                        # heatmap_complete_path = path + f'/{expl_method}/heatmap_{image_index}_batch_{batch_index}.jpg'
                        heatmap_name = files(path)
                        heatmap_complete_path = path + f'/{heatmap_name}'
                        print(os.path.exists(heatmap_complete_path))
                        print('Complete path:')
                        print(heatmap_complete_path)
                        files_heatmap_gif.append(heatmap_complete_path)
                        ax_h[math.floor(idx/n_cols), (idx-(math.floor(idx/n_cols)*n_cols))].imshow(mpimg.imread(heatmap_complete_path))

                        if expl_method == 'gradcam':
                            blended_heatmap_complete_path = path + f'/blended_{heatmap_name}'
                            files_blended_heatmap_gif.append(blended_heatmap_complete_path)
                            ax_b[math.floor(idx/n_cols), (idx-(math.floor(idx/n_cols)*n_cols))].imshow(mpimg.imread(blended_heatmap_complete_path))
                    except:
                        pass

                image_id = heatmap_name.split('_')[-1].split(".")[0]
                
                dir = os.path.join(args.output_dir, 'visualizations', f'{model}',f'{expl_method}') 
                if not os.path.exists(dir):
                    os.mkdir(dir)

                fig_h.suptitle(expl_method)
                fig_h.savefig(os.path.join(args.output_dir, 'visualizations', f'{model}',f'{expl_method}',  f'heatmap_global_{model}_{expl_method}_{cls}_{image_id}.pdf'))
                if expl_method == 'gradcam':
                    fig_b.suptitle(expl_method)
                    fig_b.savefig(os.path.join(args.output_dir, 'visualizations', f'{model}',f'{expl_method}',  f'blended_heatmap_global_{model}_{expl_method}_{cls}_{image_id}.pdf'))
                
                try:
                    frames = [Image.open(image) for image in files_heatmap_gif]
                    frame_one = frames[0]
                    frame_one.save(os.path.join(args.output_dir, 'visualizations', f'{model}', f'{expl_method}', f'heatmap_global_{model}_{expl_method}_{cls}_{image_id}.gif'), format="GIF", append_images=frames, save_all=True, duration=550, loop=0)

                    if expl_method == 'gradcam':
                        frames = [Image.open(image) for image in files_blended_heatmap_gif]
                        frame_one = frames[0]
                        frame_one.save(os.path.join(args.output_dir, 'visualizations',  f'{model}', f'{expl_method}', f'blended_heatmap_global_{model}_{expl_method}_{cls}_{image_id}.gif'), format="GIF", append_images=frames, save_all=True, duration=550, loop=0)
                except:
                    pass
                    plt.close('all')


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Explaination Visualizations", add_help=add_help)

    parser.add_argument("--expl-method", default=None, type=str, help="Chosen explainability method")
    parser.add_argument("--model", default="resnet18", type=str, help="Chosen model for which perform explainations")
    parser.add_argument("--image-index", default=None, type=int, help="indexes list of images to which compute explaination")
    parser.add_argument("--batch-index", default=None, type=int, help="indexes list of images to which compute explaination")
    parser.add_argument("--expl-img-path", default=None, type=str, help="indexes list of images to which compute explaination")
    parser.add_argument("--output-dir", default=None, type=str, help="indexes list of images to which compute explaination")

    return parser 

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
