import torch
from tqdm import tqdm


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def add_multiple_to_sorted_list(values, sorted_list, max_size, do_sort):
    sorted_list.extend(values)  # Add all values to the sorted list
    if do_sort:
        sorted_list.sort()  # Sort the list
    
    if len(sorted_list) > max_size:
        sorted_list = sorted_list[-max_size:]  # Keep the last max_size elements
    return sorted_list

def to_numpy(tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()

def evaluate(model, loader):
    torch.cuda.empty_cache()
    with torch.no_grad():
        cnt = 0; corr=0
        for img, lbl in tqdm(loader, total=len(loader), desc='Evaluating the model'):
            img = img.cuda(); lbl = lbl.cuda()
            out = model(img)
            pred = out.argmax(dim=-1)
            corr += (pred == lbl).sum().item()
            cnt += len(img)
    print(100 * corr / cnt)
    torch.cuda.empty_cache()

def set_border_color(ax, color, width=3):
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(width)

def get_subset_by_class(orig_dataset, class_subset_names):
    class_subset_indices = [orig_dataset.class_to_idx[cls_name] for cls_name in class_subset_names]

    sample_indices = [idx for idx, target in enumerate(orig_dataset.targets) if target in class_subset_indices]
    # print(f"Chose {len(sample_indices)} samples for {len(class_subset_indices)} classes.")

    new_dataset = torch.utils.data.Subset(dataset=orig_dataset, indices=sample_indices)
    return new_dataset, class_subset_indices, sample_indices

from contextlib import contextmanager
import sys
from io import StringIO
@contextmanager
def silence_stdout():
    # Save the original stdout
    old_stdout = sys.stdout
    # Replace sys.stdout with a StringIO object that does nothing
    sys.stdout = StringIO()
    try:
        yield
    finally:
        # Restore the original stdout
        sys.stdout = old_stdout
