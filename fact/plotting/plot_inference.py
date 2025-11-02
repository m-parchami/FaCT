
import pickle
import os
import matplotlib
import numpy as np
import torch
import matplotlib.image as mpimg

from os.path import join as ospj
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as trn
from torchvision.datasets import ImageFolder
from fact.utils import (
    eval_sae, assign_sae_hook, assign_channel_hook,
    gradient_to_image, set_border_color,
    name_map, folder_label_map,
    get_args,
    AddInverse, get_subset_by_class
)
from fact.training.saes import load_SAE

def explain_concepts(args, concept_indices, w=4,):
    stats_file = f'./plots/analysis/{args.dir_tree}/Stats-{args.config_name}'
    model_stats = torch.load(f'{stats_file}-Model.pth', map_location='cpu')
    print(f"Accuracy (logged in the file) on {model_stats['total_cnt']} samples: {model_stats['acc']: 0.2f}")
    neuron_stats = torch.load(f'{stats_file}-Neurons.pth', map_location='cpu')
    all_p_contribs = neuron_stats['all_contribs']
    all_p_presence = neuron_stats['all_presence']

    #### Where to store explanations!
    prefix = f"./plots/viz/{args.dir_tree}/{args.config_name}/Concepts"
    os.makedirs(prefix, exist_ok=True)
    
    top_concept_instances = all_p_presence.topk(k=16, dim=0).indices # Top presence per concept
    n_rows=1
        
    for p_idx in concept_indices:
        sample_indices = top_concept_instances[:, p_idx]
        _file_contribs = all_p_contribs[sample_indices][:, p_idx]
        _file_presence = all_p_presence[sample_indices][:, p_idx]
        
        # Explain the specific concept on the top-activating images
        images, explanations, returned_contribs, preds, logit_to_explain, labels = explain_images(
            args, 
            sample_dict={-1: sample_indices}, just_return=p_idx, 
            custom_logit=None
        )
        
        # If the stored contribs was different from what we computed just now
        high_err = ((_file_contribs - returned_contribs[:, p_idx].cpu()).abs() > 0.1).int().sum()
        if high_err > 0: 
            print(f'!!!!! WARNING! High Error from what was written in stats-file! for {high_err} instacnes: '+\
                f'Max {(_file_contribs - returned_contribs[:, p_idx].cpu()).abs().max()}')
        
        fig, axes = plt.subplots(2*n_rows, w, figsize=(w, 2*n_rows), dpi=200)
        for r in range(2*n_rows):
            for c in range(w):
                _i = ((r//2)*w)+(c%w)
                if r%2 != 0:
                    axes[r, c].imshow(explanations[_i])
                    axes[r, c].set_xlabel(f"{name_map[labels[_i].item()].split(',')[0].capitalize()[:15]}", fontsize=7);
                else:
                    axes[r, c].imshow(images[_i, :3].moveaxis(0, -1).detach().cpu())
        
        for ax in axes.flatten(): 
            ax.set_xticks([]); ax.set_yticks([]);
        
        _fname = f"{prefix}/P{p_idx}-ImageExp.png"
        plt.subplots_adjust(hspace=0.01)
        plt.savefig(_fname, bbox_inches='tight')
        print(f'Checkout {_fname}')
        plt.cla(); plt.close();

def explain_images(
        args, sample_dict, just_return=None, custom_logit=None,
    ):
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    params = {'font.size': 24, 
            'font.family': 'sans-serif', 'font.serif': 'Computer Modern Sans serif',
            'text.usetex': False, 'text.latex.preamble': 
                "\n".join([
                r"\usepackage{lmodern}", r'\usepackage{amssymb}', r'\usepackage{amsmath}',
                r'\usepackage{wasysym}', r'\usepackage{xcolor}', r'\usepackage{pifont}',
                r'\usepackage{xspace}']),
            }
    plt.rcParams.update(params)

    transform=trn.Compose([
                trn.Resize(256), trn.CenterCrop(224),
                trn.ToTensor(), AddInverse(),
            ])
    full_val_dataset = ImageFolder(
        '/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized',
        transform=transform
    )
    
    dataset = full_val_dataset

    torch.hub.list('B-cos/B-cos-v2')
    model = torch.hub.load('B-cos/B-cos-v2', args.arch, pretrained=True)
    model.eval(); model.cuda()
    torch.cuda.empty_cache()
    
    if args.method == 'channels':
        sae_model = None
        hook = assign_channel_hook(model, args.layer)
    else:
        sae_model, _, _ = load_SAE(args)
        hook = assign_sae_hook(model, args.layer, sae_model)
    
    topk = 1 if just_return is not None else 16
    topk_offset = 0
    
    all_p_indices_plotted = []
    for cls_idx, sample_indices in sample_dict.items():
        if cls_idx != -1:
            cls = list(folder_label_map.keys())[cls_idx]
            dataset, _, _ = get_subset_by_class(full_val_dataset, [cls])

        dataset_selected = torch.utils.data.Subset(dataset, sample_indices)
        val_loader = DataLoader(
            dataset_selected, batch_size=min(16, len(sample_indices)),
            shuffle=False, drop_last=False, num_workers=1,
        )
        
        with model.explanation_mode():
            ret_dict = eval_sae( 
                val_loader, model=model, sae_model=sae_model,
                force_label=False, # Explain the predicted logit (not the image label)
                custom_logit=custom_logit, # Explain a custom logit
                one_batch_only=True, # Keep activations attached to the graph (for backward passes below)
            )
            
            orig_weights = ret_dict['orig_weights']
            logits = ret_dict['logits'];
            logits_to_explain = ret_dict['logits_to_explain']
            images = ret_dict['images'];
            all_contribs = ret_dict['p_contribs']
            all_presence = ret_dict['p_presence']
            B = len(images)
            preds = logits.argmax(dim=-1).cpu()
            labels = ret_dict['labels']
            
            the_images = images.detach().cpu()
            grads = torch.zeros((len(images), 1+topk+1, *images.shape[1:]))
            explanations = np.zeros((len(images), 1+topk+1, 224, 224, 4))

            if just_return is not None:
                top_indices = (torch.ones(B, 1) * just_return).long()
            else:
                top_indices = all_contribs.topk(k=topk_offset + topk, dim=1).indices[:, topk_offset:]
            all_p_indices_plotted += [top_indices]
            plotted_contribs = torch.zeros_like(all_contribs[:, 0])
            for top_idx in range(topk):
                p_indices = top_indices[:, top_idx]
                act_to_explain = all_presence[range(B), p_indices]
                # # If you want to visualize the "contribs" (Appendix H.2)
                # act_to_explain = all_contribs[range(B), p_indices] 
                act_to_explain.sum(dim=0).backward(
                    inputs=[images], create_graph=False, retain_graph=True
                )
                for i_idx in range(B):
                    grads[i_idx, top_idx] = images.grad[i_idx].detach().cpu();
                images.grad = None
                plotted_contribs += all_contribs[range(B), p_indices]

            # Store the Logit's explanation
            grads[:, -1] = orig_weights 
            
            # Explain other (non-plotted) concepts together
            other_concept_contribs = all_contribs.sum(dim=1) - plotted_contribs
            other_concept_contribs.sum(dim=0).backward(
                inputs=[images], create_graph=False, retain_graph=True
            )
            grads[:, -2] = images.grad.detach().cpu()
            images.grad = None

        # Compute the explanations (i.e visualize the W(x) at input-level)
        # Same way as in the original B-cos paper
        for i in range(len(images)):
            for p_idx in range(topk+2):
                explanations[i, p_idx] = gradient_to_image(
                    the_images[i], grads[i, p_idx],
                    rescale='default', smooth=3,
                )
        
        if just_return is not None:
            assert len(sample_dict) == 1
            return images, explanations[:, 0], all_contribs, preds, logits_to_explain.detach().cpu(), labels

        #####################################

        fig, axes = plt.subplots(
            len(sample_indices), 2+topk+1, 
            figsize=np.array((2+topk+1, len(sample_indices)+0.2))*4, dpi=300)
        if len(sample_indices) == 1:
            axes = axes[None]
        for ax in axes.flatten(): ax.set_xticks([]); ax.set_yticks([]);

        for i in range(len(images)):
            save_path = f'./plots/viz/{args.dir_tree}/{args.config_name}/{cls_idx}-{sample_indices[i]}-{custom_logit}'
            os.makedirs(save_path, exist_ok=True)
            # Plot the Image with its label and predicted class
            _img = images[i, :3].moveaxis(0, -1).detach().cpu()
            axes[i, 0].imshow(_img)
            axes[i, 0].set_xlabel(f"Label {labels[i]} {name_map[preds[i].item()].split(',')[0][:12]}\n"+\
                                f"Pred {preds[i].item()} {name_map[preds[i].item()].split(',')[0][:12]}\n"+\
                                f"Explaining {custom_logit}" if custom_logit is not None else "")
            set_border_color(axes[i, 0], 'red' if preds[i] != cls_idx else 'green')
            
            axes[i, 1].imshow(explanations[i, -1])
            axes[i, 1].set_xlabel(f"Logit {logits_to_explain[i].item():0.1f}")
            set_border_color(axes[i, 1], 'purple')
            
            # Also save in files
            mpimg.imsave(f'{save_path}/Image.png',_img.numpy())
            mpimg.imsave(f'{save_path}/Orig.png',explanations[i, -1])

            # Top K prototypes for the class
            p_contribs = all_contribs[i].detach()
            for top_idx in range(topk):
                p_idx = top_indices[i, top_idx]
                contribution = 100 * p_contribs[p_idx] / p_contribs.clip(0, None).sum(0)
                axes[i, 2+top_idx].set_xlabel(f"P{p_idx}: {contribution.item():0.1f}%")
                axes[i, 2+top_idx].imshow(explanations[i, top_idx])
                # Also save in files
                mpimg.imsave(f'{save_path}/{p_idx}.png',explanations[i, top_idx])

            # Report and visualize the contribution of other concepts (outside top-k contributing ones)
            other_contribs = (other_concept_contribs[i] / logits_to_explain[i]).item()
            if other_contribs > 0.01:
                axes[i, -1].imshow(explanations[i, -2])
            axes[i, -1].set_xlabel(f"Other Concepts {100*other_contribs:0.1f} %")
            set_border_color(axes[i, -1], 'brown')

        plt.savefig(f'./plots/viz/{args.dir_tree}/{args.config_name}/Overview.png', bbox_inches='tight')
        print(f"Checkout ./plots/viz/{args.dir_tree}/{args.config_name}/Overview.png")

    hook.remove(); 
    all_p_indices_plotted = torch.concat(all_p_indices_plotted, dim=0)
    return all_p_indices_plotted
 
def plot_local_and_global(args, sample_dict):
    os.makedirs(f"./plots/viz/{args.dir_tree}/{args.config_name}/viz-mixed/", exist_ok=True)
    for cls_idx, sample_indices in sample_dict.items():
        for sample_idx in sample_indices:
            viz_dir = f"./plots/viz/{args.dir_tree}/{args.config_name}/{cls_idx}-{sample_idx}-None"
            concepts_dir = f"./plots/viz/{args.dir_tree}/{args.config_name}/Concepts"
            concepts = sorted(os.listdir(viz_dir))

            for concept_file in concepts:
                if concept_file in ['Image.png', 'Orig.png']:
                    continue

                concept_idx = os.path.splitext(concept_file)[0]
                viz_img = Image.open(ospj(viz_dir, concept_file))
                proto_img = Image.open(ospj(concepts_dir, f"P{concept_idx}-ImageExp.png"))
                
                fig = plt.figure(figsize=(5, 2), dpi=180)
                gs = GridSpec(1, 2, width_ratios=[1, 2])
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1])
                axes = np.array([ax1, ax2])
                
                axes[0].imshow(viz_img);
                axes[1].imshow(proto_img);
                for spine in axes[0].spines.values():
                    spine.set_color('black'); spine.set_linewidth(1)
                axes[1].axis('off')
                
                for ax in axes.flatten(): 
                    ax.set_xticks([]); ax.set_yticks([]);
                
                plt.subplots_adjust(wspace=0.01)
                fname = f"./plots/viz/{args.dir_tree}/{args.config_name}/viz-mixed/{cls_idx}-S{sample_idx}-P{concept_idx}.png"
                plt.savefig(fname, bbox_inches='tight')
                plt.cla(); plt.close();
                print(f'Checkout {fname}')

if __name__ == '__main__':
    args = get_args()
    
    ### These are a bunch of concept indices from DenseNet Norm 5 (pen-ultimate) layer
    ### Some of which have been shown in the paper and the Appendix
    explain_concepts(args, concept_indices=[391, 409, 764, 1340, 1554, 1825, 2527,
                    4159, 4178, 7014, 13249, 16097, 2068, 16238, 266, 4383, 2096, 551, 16077, 14812,
                    12853, 4798, 7010, 9467, 10587, 11720, 16246], w=6)
    
    ### Visualize 4 samples for some classes. First see which concepts activate and contribute (explain_images)
    ### Then visualize those concepts (explain_concepts). You can then put both together with (plot_local_and_global)
    # sample_dict = {
    #     779: [0,1,2,3], # School Bus
    #     # 511: [0,1,2,3], # Convertible
    # }
    # all_p_indices_plotted = explain_images(args, sample_dict=sample_dict)
    # concept_indices = all_p_indices_plotted.flatten().unique().cpu().tolist()
    # explain_concepts(args, concept_indices=concept_indices, w=4)
    # plot_local_and_global(args, sample_dict) # Must have done the explain-images and explain_concepts beforehand

