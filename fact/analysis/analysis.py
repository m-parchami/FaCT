import os
import random
import matplotlib
import numpy as np
import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as trn
from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize

from fact.utils import (
    AddInverse,  
    IMAGENET_MEAN, IMAGENET_STD,
    assign_channel_hook, assign_sae_hook,  eval_sae,
    get_args,
)
from fact.training.saes import load_SAE

def analysis(args):    
    plt.rcParams.update({'font.size': 18})
    torch.manual_seed(623); random.seed(623); np.random.seed(623);
    
    transform=trn.Compose([
            trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(),
            AddInverse() if args.bcos_layers else Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    full_val_dataset = ImageFolder( 
        root='/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized',
        transform=transform
    )
    total_cnt = len(full_val_dataset)
    print(f"Total Number of Samples: {total_cnt}")
        
    if args.bcos_layers:
        torch.hub.list('B-cos/B-cos-v2')
        model = torch.hub.load('B-cos/B-cos-v2', args.arch, pretrained=True)
    else:
        assert args.arch == 'std_resnet50'
        from torchvision.models import ResNet50_Weights, resnet50 as standard_r50
        model = standard_r50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    model.eval(); model.cuda()
    if args.method in ('channels', 'crp'):
        sae_model = ckpt_path = None; train_log = {}; # For consistency in log files
        hook = assign_channel_hook(model, args.layer)
    else:
        sae_model, ckpt_path, train_log = load_SAE(args)
        hook = assign_sae_hook(model, args.layer, sae_model)
    
    # Store in two different dicts (for faster loading)
    model_stats, neuron_stats = dict(), dict()
    
    prefix = f'./plots/analysis/{args.dir_tree}'
    os.makedirs(prefix, exist_ok=True)
    dump_name = f'{prefix}/Stats-{args.config_name}'
    
    torch.cuda.empty_cache()
    val_loader = DataLoader(full_val_dataset, batch_size=64,
        shuffle=False, drop_last=False, num_workers=5,
    )
    all_contribs, all_logits, \
        correct_indices, all_presence, l2_errors = eval_sae(
            val_loader, model=model, sae_model=sae_model,
            one_batch_only=False, force_label=False,
        )
    hook.remove();
    
    neuron_stats['all_contribs'] = all_contribs
    neuron_stats['all_presence'] = all_presence
    neuron_stats['all_logits'] = all_logits
    
    model_stats['train_log'] = train_log
    model_stats['ckpt'] = ckpt_path
    model_stats['total_cnt'] = total_cnt
    model_stats['correct_indices'] = correct_indices
    model_stats['Image-L0'] = (all_presence.abs() > 0).float().sum(1).mean(0).item()
    model_stats['L2_error'] = torch.std_mean(l2_errors.flatten(), dim=0)

    acc = 100 * len(correct_indices) / total_cnt
    model_stats['acc'] = acc
    print(f"Final Accuracy on {total_cnt} samples {acc:0.2f}")

    pos_contribs = all_contribs.clip(0, None)
    all_pos_contribs_portion = (100 * pos_contribs / pos_contribs.sum(dim=1, keepdim=True))
    sorted_indices = all_pos_contribs_portion.sort(dim=1, descending=True).indices
    cumsum = torch.cumsum(torch.gather(all_pos_contribs_portion, dim=1, index=sorted_indices), dim=1)
    
    # Additional stats: num concepts to get to 80% of the positive contribs
    coverage_th = 80
    indices = (cumsum >= coverage_th).float().argmax(dim=1) + 1
    model_stats[f'{coverage_th}-PosCover-L0'] = indices.float().mean().item()
    
    # Additional stats: num concepts that contribute to >85% of samples
    always_alive_th = 0.85 
    always_alive_mask = (all_contribs.abs() > 0).float().mean(0) > always_alive_th
    always_alive_overall_contrib = all_pos_contribs_portion[:, always_alive_mask].abs().sum(dim=1).mean(dim=0)
    alive_percentage = 100*(all_contribs.abs().sum(0) > 0).float().mean(0)
    model_stats['alive_percentage'] = alive_percentage
    model_stats['always_alive_th'] = always_alive_th
    model_stats['always_alive_mask'] = torch.where(always_alive_mask)[0]
    model_stats['coverage_cumsum'] = torch.std_mean(cumsum, dim=0)
    
    desc=f"""
        Nr Concepts: {args.nr_concepts} Sparsity: {args.sparsity}
        Accuracy: {acc:0.2f}%
        Image-L0: {model_stats[f'Image-L0']:0.1f}. L0 @ {coverage_th}% PosCover: {model_stats[f'{coverage_th}-PosCover-L0']:0.1f}
        Alive Concepts: {alive_percentage:0.1f}%
        Always (>{100*always_alive_th}%) contrib: {always_alive_mask.int().sum(0)}
        together contrib: {always_alive_overall_contrib:0.2f}% 
    """
    print(desc)
    
    torch.save(model_stats, f'{dump_name}-Model.pth')
    torch.save(neuron_stats, f'{dump_name}-Neurons.pth')
    print(f'Model and Neurons stats saved as {dump_name}-(Model/Neurons).pth')
    

if __name__ == '__main__':
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    params = {'font.size': 22, 
            'font.family': 'sans-serif', 'font.serif': 'DejaVu Sans',
            }
    plt.rcParams.update(params)

    args = get_args()    
    print(args)
    analysis(args)