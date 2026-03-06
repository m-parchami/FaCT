import re
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm.rich import tqdm

import torch
from torchvision import transforms as trn
from torchvision.transforms.functional import resize
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_raw_encoding(attr_pos, hr_feat):
    """
    Compute attribution-weighted spatial average of features.
    Args:
        attr_pos (FloatTensor): (B, 1, H, W) spatial attribution map.
        hr_feat (FloatTensor): (B, E, H, W) feature map.

    Returns:
        FloatTensor: (B, E) weighted feature encoding.
    """
    return ((attr_pos * hr_feat).sum((-2, -1)) / attr_pos.sum((-2, -1)))

def weighted_pairwise_cosine_similarity(x: np.ndarray, w: np.ndarray):
    """
        Compute weighted pairwise cosine similarity between vectors.

        Args:
            x (np.ndarray): Input vectors. (B, E)
            w (np.ndarray): Weights. (B,)

        Returns:
            np.ndarray: Weighted cosine similarity matrix.
    """
    assert x.shape[0] == w.shape[0] and x.ndim == 2 and w.ndim == 1
    w = w / w.sum()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T  # (N, N)
    weight_matrix = w[:, None] * w[None, :]
    mask = ~np.eye(len(x), dtype=bool) # Exclude the diagonal (self-similarity)
    return ((sim[mask] * weight_matrix[mask]).sum() / weight_matrix[mask].sum()).item()

def load_top_activations(concept_dir):
    """
        Load top-activating samples for a concept.

        Returns:
            list[tuple[str, float, FloatTensor]]:
                (image_path, activation_value, attribution_map (H, W))
    """
    pattern = re.compile(r"id_\d+_act_([0-9.]+)\.jpg")

    out = [] # List of tuples (img_path, act_value, act_map)
    for img_path in concept_dir.glob("id_*_act_*.jpg"):
        m = pattern.match(img_path.name)
        if m is None:
            raise RuntimeError(f"{img_path=}")

        act_value = float(m.group(1))

        stem = img_path.name.split("_act_")[0]

        actmap_path = img_path.with_name(stem + "_actmap.pt")
        actmap = torch.load(actmap_path)
        
        out.append((str(img_path), act_value, actmap))

    out.sort(key=lambda x: x[1])
    return out

def c2_score_minimal(method_name, concept_dump_path, min_entry_count=0):
    consistency_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    consistency_model.cuda(); consistency_model.eval(); 
    consistency_model.compile()
    def fw(images):
        # Should return (B, C, H, W) features. Adjust this if you change your model above.
        feats = consistency_model.forward_features(images)["x_norm_patchtokens"]
        return feats.reshape(len(images), 16, 16, -1).moveaxis(-1, 1)
        
    upsampler = torch.hub.load('wimmerth/anyup', 'anyup')
    upsampler.cuda(); upsampler.eval()
    
    transform = trn.Compose([trn.ToTensor(), trn.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
    scores = dict() # dict(concept_id -> score value)
    for concept_idx, concept_dir in enumerate(tqdm(sorted(concept_dump_path.glob("concept_*")))):
        concept_id = concept_dir.name

        ### Read the data for the concept
        tuples = load_top_activations(concept_dir)
        if len(tuples) < min_entry_count:
            print(f'skipping {concept_id} with {len(tuples)} entries < {min_entry_count}')
            continue
            
        images = []; actmaps = []; act_vals = []
        for img_path, act, actmap in tuples:
            images += [transform(Image.open(img_path).convert("RGB"))]
            actmaps += [actmap]
            act_vals += [act]
        images = torch.stack(images, dim=0) # (B, 3, H, W)
        actmaps = torch.stack(actmaps, dim=0)[:, None] # (B, 1, H', W')
        act_vals = torch.Tensor(act_vals) # (B,)

        ### Compute DINOv2 features
        with torch.no_grad():
            images = images.cuda()
            lr_feats = fw(images) # DINOv2 forward
            hr_feats = upsampler(images, lr_feats) # Upsample the features
            actmaps = resize(actmaps.clip(0, None), images.shape[-2:]) # Upsample attribution maps

        ### Compute a single DINO encoding per image (based on attribution map)
        encodings = get_raw_encoding(actmaps.cuda(), hr_feats).detach().cpu().float()
        
        ### Compute a pairwise similarity of encodings
        consistency_score = weighted_pairwise_cosine_similarity(encodings, act_vals)
        assert concept_id not in scores
        scores[concept_id] = consistency_score
        
        ### Just intermediate checkpoints
        if (concept_idx+1) % 500 == 0: 
            torch.save(scores, f'{method_name}-scores.pt')
            
    torch.save(scores, f'{method_name}-scores.pt')
    return scores

    
if __name__ == '__main__':
    
    scores = c2_score_minimal(
        method_name='dummy_name', # for saving the file name
        concept_dump_path=Path('dummy_dir'), # path to root dir of your concepts
        min_entry_count = 0 # only include concepts with at least these many activating images
    )
    
    print(
        f"""
          Saved the score. 
          C2-Score value for {len(scores)} Concepts is {np.mean(list(scores.values())).item():0.3f}
        """
    )
    print(scores) # For the dummy example, second concept should have a higher score