import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib
import numpy as np

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as trn
from torchvision.datasets import ImageFolder
from torchvision.transforms import Normalize
from torchvision.transforms.functional import resize
import random
from tqdm.rich import tqdm
from fact.utils import (
    AddInverse,
    assign_channel_hook, assign_sae_hook,
    IMAGENET_MEAN, IMAGENET_STD,
    features_and_weights_sae,
    get_args
)
from contextlib import nullcontext
from fact.training.saes import load_SAE

from torchvision.models import resnet50 as standard_r50
from torchvision.models import ResNet50_Weights
from craft.craft.craft_torch import Craft

method_labels = {
    'BiasFreeTopK': 'Our Concepts',
    'channels': 'B-cos Channels',
    'craft': 'CRAFT Concepts',
    'crp': 'CRP Concepts',
}
fmt = lambda c: np.asarray(c)/255

from loftup.featurizers import get_featurizer

def plot_consistencies(concept_stats, random_baseline, *, min_cnt=None, ax=None, just_return = False):
    keys, counts, consistencies, _ = concept_stats.T

    if min_cnt is not None:
        enough_indices = torch.where(counts >= min_cnt)[0]
        print(f'Considering only concepts with >={min_cnt} samples {len(enough_indices)} of {len(counts)}')
        consistencies, keys, counts = consistencies[enough_indices], keys[enough_indices], counts[enough_indices]
        
    mean_consistency = consistencies.mean(0).item()
    diff = mean_consistency - random_baseline
    if just_return:  
        return diff
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.tick_params(labelleft=True)
    ax.set_yticks(ticks=[15, 30, 45], labels=[15, 30, 45], fontsize=12)

    n_bins = 50
    ax.set_xticks(ticks=[0, 0.5, 1.0], labels=[0, 0.5, 1.0], fontsize=12);
    counts, bins = np.histogram(consistencies, bins=n_bins, range=(-1, 1), density=False)
    
    counts = 100 * counts / counts.sum()
    ax.bar(
        bins[:-1], counts, width=np.diff(bins), align='edge',
        alpha=0.8, edgecolor='black', linewidth=0.7, color=fmt([103,169,207])
    )
    ax.axvline(mean_consistency, color=fmt([84,39,136]), linestyle='--', label='Mean')
    ax.axvline(random_baseline, color='red', linestyle='-', label='Random Baseline')

    y = 20
    arrow_props = dict(arrowstyle='<->', color='black')
    ax.annotate('', xy=(random_baseline, y), xytext=(mean_consistency, y), arrowprops=arrow_props)
    better = max(mean_consistency, random_baseline)
    ax.text(
        better + 0.05, y - 0.1,
        '$\\Delta_{\\text{rand}}='+f'{diff:0.2f}$', ha='left', va='center', color='green',
        fontsize=12
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 46)
    ax.set_ylabel('Percentage of Concepts', fontsize=16)
     
def get_raw_encoding(attr_pos, hr_feat):
    return ((attr_pos * hr_feat).sum((-2, -1)) / attr_pos.sum((-2, -1)))

def weighted_pairwise_cosine_similarity(x: np.ndarray, w: np.ndarray):
    assert x.shape[0] == w.shape[0] and x.ndim == 2 and w.ndim == 1
    w = w / w.sum()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T  # (N, N)
    weight_matrix = w[:, None] * w[None, :]
    mask = ~np.eye(len(x), dtype=bool)
    return ((sim[mask] * weight_matrix[mask]).sum() / weight_matrix[mask].sum()).item()

def compute_scores(dump_name, per_class=False):
    if not os.path.exists(f'{dump_name}-ConceptStats.pth'):
        files_list = [f for f in glob.glob(f'{dump_name}-RawEncodings*.npy')]
        print("Reading all features from:\n\t{}".format("\n\t".join(files_list)))
        arrs = np.vstack([np.load(file_path).astype(np.float64) for file_path in files_list])
        arrs = arrs[np.argsort(arrs[:, 0])]
        keys, presences, dataset_indices, labels, pixels = arrs.T[:5]
        encodings = arrs[:, 5:]
        unique_ids, start_idx, counts = np.unique(keys, return_index=True, return_counts=True)
        print(f"{unique_ids.shape=} {keys.shape=} {presences.shape=} {dataset_indices.shape=} \
            {labels.shape=} {pixels.shape=} {encodings.shape=}")
        
        
        mean_features = encodings[keys==-1]
        assert mean_features.shape[0] == 50_000
        print(mean_features.shape)
        dataset_mean = mean_features.mean(0, keepdims=True)
        ### Optionally save/load the dataset mean here
        # np.save('./plots/dino_mean.npy', dataset_mean) 
        # dataset_mean = np.load('./plots/dino_mean.npy') 
            
        concept_stats = []; random_stats = []; to_ignore = []; 
        for i, (key, start, count) in enumerate(zip(unique_ids, start_idx, counts, strict=True)):
            if key == -1:  continue # Mean vectors 
            concept_encodings = encodings[start : start + count]
            concept_presences = presences[start : start + count]
            concept_pixels = pixels[start: start+count]
            assert np.all(concept_presences > 0)
            if count < 2:        
                to_ignore += [key]
                continue
            
            concept_encodings -= dataset_mean # This is especially important if using LoftUP Upsampler
            consistency_score = weighted_pairwise_cosine_similarity(concept_encodings, concept_presences)
            
            spatial_size = concept_pixels.mean(0).item()
            row = torch.Tensor([key, count, consistency_score, spatial_size])
            if key == -2 or (per_class and key%100 == 99):
                random_stats += [row]
            else:
                concept_stats += [row]
        
        print(f"Warning! Ignoring {len(to_ignore)} / {len(unique_ids)} keys")
        concept_stats = torch.stack(concept_stats, dim=0)
        random_stats = torch.stack(random_stats, dim=0)
        random_stats = random_stats.mean(0) # When having multiple (per-class) random baselines
            
        print(f"Saving concept stats to {dump_name}-ConceptStats.pth")
        torch.save([concept_stats, random_stats], f'{dump_name}-ConceptStats.pth')
    else:
        concept_stats, random_stats = torch.load(f'{dump_name}-ConceptStats.pth', map_location='cpu')
        # E.g. you can sample concept indices from different ranges here and plot them using 
        # fact.plotting.plot_inference
        
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_consistencies(concept_stats, random_stats[2].item(), ax=ax, min_cnt=5)
    plt.savefig(f"{dump_name}-ConceptStats.pdf", bbox_inches='tight')
    print(f"checkout {dump_name}-ConceptStats.pdf")
    plt.cla(); plt.close();

def concept_stats(args):
    chunk_size = 60; 
    
    prefix = f'./plots/consistency/{args.dir_tree}'
    os.makedirs(prefix, exist_ok=True)
    dump_name = f'{prefix}/Stats-Top{args.quantile_per_concept}PerConcept-{args.config_name}'

    if not os.path.exists(f'{dump_name}-RawEncodings-Last.npy'):
        print('Did not find the last -RawEncoding.npy checkpoint. Computing the values!')
        transform=trn.Compose([
            trn.Resize((256,)), trn.CenterCrop(224),
            trn.ToTensor(), AddInverse(),
        ])

        val_dataset = ImageFolder(
            root='/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized',
            transform=transform,  
        )
        torch.hub.list('B-cos/B-cos-v2')

        model = torch.hub.load('B-cos/B-cos-v2', args.arch, pretrained=True)
        model.eval(); model.cuda()
        
        if args.method == 'channels':
            sae_model = None
            hook = assign_channel_hook(model, args.layer)
        else:
            sae_model, _, _ = load_SAE(args)
            hook = assign_sae_hook(model, args.layer, sae_model, with_error=False)
        
        featurizer_class = "dinov2"; torch_hub_name = "loftup_dinov2s"
        consistency_model, _, _ = get_featurizer(featurizer_class)
        consistency_model.cuda(); consistency_model.eval()
        upsampler = torch.hub.load('andrehuang/loftup', torch_hub_name, pretrained=True)
        upsampler.cuda(); upsampler.eval()
        # Optionally you can switch the upsampler model 
        # (you may need to change the forward call below as well)
        # upsampler = torch.hub.load('wimmerth/anyup', 'anyup')
        
        std_normalize = Normalize(IMAGENET_MEAN, IMAGENET_STD)
        
        concept_encodings = []
        
        torch.manual_seed(512); random.seed(512); np.random.seed(512);
        torch.cuda.empty_cache()
        all_p_presence = get_dataset_stats(args)
        images_to_concepts = get_indices_to_explain(all_p_presence, quantile=args.quantile_per_concept)
        with model.explanation_mode():
            batch_size = 32
            loader = DataLoader(
                val_dataset, batch_size=batch_size, drop_last=False, num_workers=10,
                shuffle=False,
            )
            for batch_idx, (images, labels) in tqdm(enumerate(loader), 'Collecting features', total=len(loader)):        
                next_ckpt = f'{dump_name}-RawEncodings-B{((batch_idx//chunk_size)+1)*chunk_size - 1}.npy'
                if os.path.exists(next_ckpt):
                    print(f'skipping {batch_idx} because {next_ckpt} exists')
                    continue
                b, c, h, w = images.shape
                images = images.cuda()
                with torch.no_grad():
                    _img = std_normalize(images[:, :3])
                    hr_feats = upsampler(consistency_model(_img), _img)
                    # hr_feats = upsampler(_img, consistency_model(_img)) # in case of AnyUp
                
                ret_dict = features_and_weights_sae(
                    images, model, sae_model,
                    labels=labels, force_label=True, with_graph=True)
                p_presence = ret_dict.pop('p_presence')
                images = ret_dict.pop('images')
                
                ################## Take topk_per_concept per concept
                global_indices = range(batch_idx*loader.batch_size, batch_idx*loader.batch_size + b)
                concepts_to_explain = [images_to_concepts[global_idx] for global_idx in global_indices]
                per_img_counts = [len(c_indices) for c_indices in concepts_to_explain]
                max_count = max(per_img_counts)
                for rank_idx in range(max_count):
                    rank_local_indices = torch.Tensor([local_idx for local_idx, c_indices in enumerate(concepts_to_explain)
                                                            if len(c_indices) > rank_idx]).long()
                    rank_concept_indices = torch.Tensor([c_indices[rank_idx] for c_indices in concepts_to_explain
                                                            if len(c_indices) > rank_idx]).long()

                    images.grad = None
                    p_presence[rank_local_indices, rank_concept_indices].sum(0).backward(
                        inputs=[images], retain_graph=True, create_graph=False
                    )
                    attr_pos = (images.grad[rank_local_indices] * images.detach()[rank_local_indices]).sum(1, keepdim=True).clip(0, None)
                    images.grad = None
                    encodings = get_raw_encoding(attr_pos, hr_feats[rank_local_indices]).detach().cpu().float()
                    for i__, (local_idx, c_idx, attr, encoding) in enumerate(zip(rank_local_indices, rank_concept_indices, attr_pos, encodings, strict=True)):
                        presence = p_presence[local_idx, c_idx].item()
                        if presence > 0:
                            vals = torch.sort(attr.flatten(), descending=True).values
                            n_pixels = ((vals.cumsum(0) / vals.sum() >= 0.8).nonzero().min() + 1).item()
                            entry = dict(
                                presence = presence, encoding = encoding.tolist(), n_pixels = n_pixels,
                                img_idx = global_indices[local_idx],
                                img_label = labels[local_idx].item(),
                            )
                            concept_encodings += [[c_idx.item(), entry['presence'], entry['img_idx'],
                                    entry['img_label'], entry['n_pixels'], *entry['encoding']]]
                            
                    encodings = encoding = attr_pos = presence = None
                p_presence = None
                ########################
                
                
                ###### Mean features as concept `-1`; Random baseline as concept `-2`
                for img_idx, feat in enumerate(hr_feats):
                    mean_encoding = feat.mean((-1, -2)).detach().cpu().float().tolist()
                    concept_encodings += [[-1, 0.0, img_idx + batch_idx * batch_size,
                                labels[img_idx].item(), 0, *mean_encoding]]
                
                    attr_pos = torch.rand((h, w)).clip(torch.rand((1,)).item(), None).cuda()
                    entry = dict(
                        presence = attr_pos.sum().item(),
                        encoding = get_raw_encoding(attr_pos, feat).detach().cpu().float().tolist(),
                        n_pixels = -1,
                        img_idx = img_idx + batch_idx * batch_size,
                        img_label = labels[img_idx].item(),
                    )
                    concept_encodings += [[-2, entry['presence'], entry['img_idx'],
                        entry['img_label'], entry['n_pixels'], *entry['encoding']]]

                ###### Checkpoint
                if (batch_idx+1) % chunk_size == 0:
                    fname =  f'{dump_name}-RawEncodings-B{batch_idx}'
                    np.save(fname, np.asarray(concept_encodings), allow_pickle=False)
                    print(f"Chunk Saved {fname}")
                    concept_encodings = []

        hook.remove();
        fname = f'{dump_name}-RawEncodings-Last'
        np.save(fname, np.asarray(concept_encodings), allow_pickle=False)
        print(f"Chunk Saved {fname}")
        concept_encodings = []
        print('Saved all the checkpoints;');
    
    compute_scores(dump_name, per_class=False)

def std_concept_stats(args):
    # make sure these don't get confused with FaCT and B-cos
    assert args.arch.startswith('std_') and \
        args.exp_name.startswith('std_') and \
        args.nr_concepts == 0 and \
        args.method in ('craft', 'crp')
    if args.method == 'craft':
        per_class = True
    batch_size=16; chunk_size=60;
    
    prefix = f'./plots/consistency/{args.dir_tree}'
    os.makedirs(prefix, exist_ok=True)
    dump_name = f'{prefix}/Stats-Top{args.quantile_per_concept}PerConcept-{args.config_name}'
    
    arch = args.arch
    
    last_file = f'{dump_name}-RawEncodings-{-1 if args.method =="crp" else 999}-Last.npy'
    if not os.path.exists(last_file):
        print('Did not find the last -RawEncoding.npy checkpoint. Computing the values!')
        transform=trn.Compose([
            trn.Resize((256,)), 
            trn.CenterCrop(224),
            trn.ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        val_dataset = ImageFolder(
            root='/scratch/inf0/user/mparcham/ILSVRC2012/val_categorized',
            transform=transform,  
        )
        assert arch == 'std_resnet50' and args.layer == 'layer4'
        std_r50 = standard_r50(weights=ResNet50_Weights.IMAGENET1K_V2)
        std_r50 = std_r50.eval().cuda()
        if args.method == 'craft':
            part1 = torch.nn.Sequential(
                std_r50.conv1, std_r50.bn1, std_r50.relu, std_r50.maxpool,
                std_r50.layer1, std_r50.layer2, std_r50.layer3, std_r50.layer4,
            )
            part2 = lambda x: std_r50.fc(std_r50.avgpool(x).flatten(1))
        
        torch.manual_seed(623); random.seed(623); np.random.seed(623);
        cls_datasets = {}
        match args.method:
            case 'craft':
                forward_context = torch.no_grad  
                for cls_idx in set(val_dataset.targets):
                    class_samples = [idx for idx, target in enumerate(val_dataset.targets)
                                    if target==cls_idx]
                    cls_datasets[cls_idx] = torch.utils.data.Subset(
                            dataset=val_dataset, indices=class_samples)                
                n_concepts_per_class = 16
                assert n_concepts_per_class < 100
                craft_checkpoints_dir = '/BS/mparcham2/work/FaCT/fact/craft/craft_dumps_all'
                # Tiny Hack: For per-class methods, we don't have images_to_concepts
                # instead we compute all concepts over all images, and 
                # then filter the quantile below before saving 
            case 'crp':
                forward_context = nullcontext
                per_class=False
                cls_datasets[-1] = val_dataset
                from crp.crp.attribution import CondAttribution
                from crp.crp.helper import get_layer_names
                from zennit.composites import EpsilonPlusFlat
                from zennit.torchvision import ResNetCanonizer
                composite = EpsilonPlusFlat([ResNetCanonizer()])
                attribution = CondAttribution(std_r50)
                layer_names = get_layer_names(std_r50, [torch.nn.Conv2d, torch.nn.Linear])
                all_p_presence = get_dataset_stats(args)
                images_to_concepts = get_indices_to_explain(all_p_presence, quantile=args.quantile_per_concept)
            
        print('Loading Dino')
        featurizer_class = "dinov2"; torch_hub_name = "loftup_dinov2s"
        consistency_model, patch_size, dim = get_featurizer(featurizer_class)
        consistency_model.cuda(); consistency_model.eval()
        upsampler = torch.hub.load('andrehuang/loftup', torch_hub_name, pretrained=True)
        upsampler.cuda(); upsampler.eval()
        # Optionally use other upsamplers
        # upsampler = torch.hub.load('wimmerth/anyup', 'anyup')
        
        concept_encodings = []
        img_idx_offset = 0
        for cls_idx, cls_dataset in cls_datasets.items():
            torch.cuda.empty_cache()
            loader = DataLoader(
                cls_dataset, batch_size=batch_size,
                shuffle=False, drop_last=False, num_workers=2
            )
            if args.method == 'craft':
                craft = Craft(
                    input_to_latent=part1, latent_to_logit=part2,
                    number_of_concepts=n_concepts_per_class, 
                    patch_size=64, batch_size=64, device='cuda'
                )
                craft.reducer = torch.load(
                    f'{craft_checkpoints_dir}/ckpt-{cls_idx}.pth',
                    weights_only=False)
                craft.W = craft.reducer.components_.astype(np.float32)
                img_idx_offset = 50 * cls_idx # Just to have global indexing of images
            
            for batch_idx, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
                next_ckpt = f'{dump_name}-RawEncodings-{cls_idx}-B{((batch_idx//chunk_size)+1)*chunk_size - 1}.npy'
                if os.path.exists(next_ckpt):
                    print(f'skipping {batch_idx} because {next_ckpt} exists')
                    continue
                b, c, h, w = images.shape
                with forward_context():
                    images = images.cuda()
                    hr_feats = upsampler(consistency_model(images), images).detach()
                    # hr_feats = upsampler(images, consistency_model(images)).detach().cpu() # If using AnyUp
                    match args.method:
                        case 'craft':
                            images_u = craft.transform(images)
                            p_presence = images_u.sum((1, 2))
                            for c_idx in range(images_u.shape[-1]):
                                batch_attr = torch.from_numpy(images_u[..., c_idx]).cuda()
                                for img_idx, (attr, hr_feat) in enumerate(zip(batch_attr, hr_feats, strict=True)):
                                    if p_presence[img_idx, c_idx] > 0:
                                        attr_pos = attr.clip(0, None)
                                        attr_pos = resize(attr_pos[None], (h, w))[0]
                                        vals = torch.sort(attr_pos.flatten(), descending=True).values
                                        n_pixels = (vals.cumsum(0) / vals.sum() >= 0.8).nonzero().min() + 1
                                        entry = dict(
                                            presence = p_presence[img_idx, c_idx].item(),
                                            encoding = get_raw_encoding(attr_pos, hr_feat).detach().cpu().float().tolist(),
                                            n_pixels = n_pixels.item(),
                                            img_idx = img_idx + batch_idx * batch_size + img_idx_offset,
                                            img_label = labels[img_idx].item(),
                                        )
                                        # store in AAABB format: AAA -> class_index BB -> concept_index
                                        _cidx = cls_idx*100 + c_idx 
                                        concept_encodings += [[_cidx, entry['presence'], entry['img_idx'],
                                                        entry['img_label'], entry['n_pixels'], *entry['encoding']]]
                        case 'crp':
                            for img_idx in range(b):
                                global_img_idx = img_idx + batch_idx * batch_size + img_idx_offset
                                sample = images[[img_idx]]
                                sample.requires_grad = True
                                concept_indices = images_to_concepts[global_img_idx]
                                conditions = [{args.layer: [c_idx]} for c_idx in concept_indices]
                                attr_offset = 0; channel_bsize = min(32, len(concept_indices))
                                hr_feats_img = hr_feats[img_idx]
                                if len(conditions) > 0:
                                    for batch_attr in attribution.generate(
                                                sample, conditions, composite, verbose=False, record_layer=layer_names,
                                                batch_size=channel_bsize, start_layer=args.layer):
                                        
                                        batch_acts = batch_attr.activations[args.layer].detach().sum((-1, -2)).cpu()
                                        batch_heatmap = batch_attr.heatmap.detach().clip(0, None).cpu()
                                        for attr_idx in range(len(batch_heatmap)):
                                            c_idx = concept_indices[attr_offset+attr_idx]
                                            attr_pos = batch_heatmap[attr_idx]
                                            vals = torch.sort(attr_pos.flatten(), descending=True).values
                                            n_pixels = (vals.cumsum(0) / vals.sum() >= 0.8).nonzero().min() + 1
                                            entry = dict(
                                                presence = batch_acts[attr_idx, c_idx].item(),
                                                encoding = get_raw_encoding(attr_pos.cuda(), hr_feats_img).detach().cpu().float().tolist(),
                                                n_pixels = n_pixels.item(),
                                                img_label = labels[img_idx].item(),
                                                img_idx = global_img_idx,
                                            )
                                            concept_encodings += [[c_idx, entry['presence'], entry['img_idx'],  
                                                    entry['img_label'], entry['n_pixels'], *entry['encoding']]]
                                        
                                        attr_offset += len(batch_heatmap)
                                        sample.grad = batch_heatmap = batch_attr = attr_pos = vals = None;
                                std_r50.zero_grad();
                                assert attr_offset == len(concept_indices)
                                batch_attr = batch_heatmap = images.grad = None;
                                torch.cuda.empty_cache()
                            images = None
                    
                    ##### Mean features as concept `-1`; Random baseline as concept `-2` (or XXX99 for per_class)
                    for img_idx, hr_feat in enumerate(hr_feats):
                        mean_encoding = hr_feat.mean((-1, -2)).float().tolist()
                        global_img_idx = img_idx + batch_idx * batch_size + img_idx_offset
                        concept_encodings += [[-1, 0.0, global_img_idx,
                                    labels[img_idx].item(), 0, *mean_encoding]]
                        
                        attr_pos = torch.rand((h, w)).clip(torch.rand((1,)).item(), None)
                        entry = dict(
                            presence = attr_pos.sum().item(),
                            encoding = get_raw_encoding(attr_pos.cuda(), hr_feat).detach().cpu().float().tolist(),
                            n_pixels = -1,
                            img_idx = global_img_idx,
                            img_label = labels[img_idx].item(),
                        )
                        _cidx = (cls_idx*100 + 99) if per_class else -2
                        concept_encodings += [[_cidx, entry['presence'], entry['img_idx'],
                                    entry['img_label'], entry['n_pixels'], *entry['encoding']]]
                    hr_feats = None
                    
                if not per_class and (batch_idx+1) % chunk_size == 0:
                    fname =  f'{dump_name}-RawEncodings-{cls_idx}-B{batch_idx}'
                    np.save(fname, np.asarray(concept_encodings), allow_pickle=False)
                    print(f"Chunk Saved {fname}")
                    concept_encodings = []
            
            if per_class:
                # Tiny Hack: filter the desired quantile for every concept
                # Go over the rows and create a new array only with top quantile
                all_encodings = np.asarray(concept_encodings)
                quantile_encodings = []
                keys, presences = all_encodings.T[[0, 1]]
                unique_ids = np.unique(keys).tolist()
                unique_ids.remove(-1) # For `mean' concept keep all the rows
                for c_idx in unique_ids: # For each concept take the quantile and store in new array
                    row_idx_to_concept = get_indices_to_explain(
                            torch.from_numpy(presences[keys == c_idx])[:, None], quantile=args.quantile_per_concept)
                    quantile_indices = [idx for idx, c_indices in row_idx_to_concept.items() if c_indices == [0]]
                    quantile_encodings += [all_encodings[keys==c_idx][quantile_indices]]
                concept_encodings = np.vstack(quantile_encodings)
            
            fname = f'{dump_name}-RawEncodings-{cls_idx}-Last'
            np.save(fname, np.asarray(concept_encodings), allow_pickle=False)
            print(f"Chunk Saved {fname}")
            concept_encodings = []

    compute_scores(dump_name, per_class=per_class)

def get_dataset_stats(args):
    neuron_stats = torch.load(f'./plots/analysis/{args.dir_tree}/Stats-{args.config_name}-Neurons.pth', map_location='cpu')
    all_p_presence = neuron_stats['all_presence']
    return all_p_presence

def get_indices_to_explain(all_p_presence, quantile, min_cnt=2, base_cnt=10):    
    image_to_concepts = {img_idx: [] for img_idx in range(len(all_p_presence))}
    for c_idx in range(all_p_presence.shape[1]):
        concept_acts = all_p_presence[:, c_idx]
        assert quantile is None or 0 < quantile < 1

        positive_only = concept_acts[concept_acts > 0]
        if len(positive_only) == 0: # Dead concept
            continue
        if quantile is None: # Consider all samples
            threshold = 0.0 
        elif len(positive_only) <= base_cnt: # Consider all samples
            threshold = 0.0 
        else: # Consider quantile quantile (at least 50 samples)     
            threshold = torch.quantile(positive_only, quantile).item() 
            threshold = min(threshold, torch.sort(positive_only, descending=True).values[base_cnt])
            
        selected_img_indices = (concept_acts > threshold).nonzero(as_tuple=True)[0].tolist()
        if len(selected_img_indices) < min_cnt:
            continue
        for img_idx in selected_img_indices:
            image_to_concepts[img_idx] += [c_idx]
        
        # print(f"{c_idx} {len(selected_img_indices)} of {len(concept_acts[concept_acts>0])} active images")
    
    # Sanity: make sure all indices have >=0 activation
    for img_idx, concept_indices in image_to_concepts.items():
        assert torch.all(all_p_presence[img_idx, concept_indices] >= 0)
        
    return image_to_concepts

if __name__ == '__main__':
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    params = {'font.size': 12, 
            'font.family': 'sans-serif', 'font.serif': 'DejaVu Sans',
            }
    plt.rcParams.update(params)

    args = get_args(); print(args);
    if args.arch.startswith('std_'):
        assert not args.bcos_layers
        std_concept_stats(args)
    else:
        assert args.bcos_layers
        concept_stats(args)
