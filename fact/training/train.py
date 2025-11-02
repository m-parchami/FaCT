import os
import random
import wandb
import torch

from torch.utils.data import DataLoader
from torchvision import transforms as trn
from torchvision.datasets import ImageFolder

from fact.utils import (
    AddInverse, SAETrainer, 
    eval_accuracy, assign_storage_hook, load_good_features,
    get_args
)

def train_imn(args):
    print(args)
    assert args.bcos_layers and args.method == 'BiasFreeTopK'

    wandb.init(
        entity="", 
        project='FaCT', 
        config=vars(args),
        name=args.config_name
    )
    transform = trn.Compose([
            trn.Resize(256), trn.CenterCrop(224), 
            trn.ToTensor(), AddInverse(),
        ])

    full_train_dataset = ImageFolder(
                root='/scratch/inf0/user/mparcham/ILSVRC2012/train',
                transform=transform
            )
    print(f"Loaded ImageNet with {len(full_train_dataset)} samples and {transform=}")

    ##### Take a 50-shot per class heldout subset
    random.seed(42)
    heldout_indices, train_indices = [], []
    for t in range(1000):
        class_samples = [idx for idx, target in enumerate(full_train_dataset.targets) if target==t]
        heldout = random.sample(class_samples, 50)
        train = [idx for idx in class_samples if idx not in heldout]
        heldout_indices += heldout
        train_indices += train
    full_val_dataset = torch.utils.data.Subset(dataset=full_train_dataset, indices=heldout_indices)
    full_train_dataset = torch.utils.data.Subset(dataset=full_train_dataset, indices=train_indices)
    print(f"{len(heldout_indices)=} {len(train_indices)=} {hash(tuple(heldout_indices))=}")
    print(f"{len(full_train_dataset)=}  {len(full_val_dataset)=}")
    
    ##### Get the initial model and evaluate it
    torch.hub.list('B-cos/B-cos-v2')
    model = torch.hub.load('B-cos/B-cos-v2', args.arch, pretrained=True)
    model.eval(); model.cuda()
    
    # Use this to see layer names (for selecting the hook layer)
    # print(model); print('\n'.join([name for name, _ in model.named_modules()]))
    
    torch.cuda.empty_cache()
    loading_batch_size = 260
    loader = DataLoader(full_train_dataset, batch_size=loading_batch_size, 
        shuffle=True, drop_last=False, num_workers=10,
    )
    val_loader = DataLoader(full_val_dataset, batch_size=128, 
        shuffle=False, drop_last=False, num_workers=10,
    )
    # print('Checking original model accuracy on held-out set!')
    # correctness = eval_accuracy(val_loader, model)
    # acc = correctness.float().mean()
    # print(f"Accuracy over {len(correctness)} Samples: {100*acc:0.2f}")
    
    
    ##### Gather the Features
    torch.cuda.empty_cache()
    sample_cnt = None
    if args.precomputed_feats is not None:
        print(f"Loading from the disk: {args.precomputed_feats.split('/')[-1]}")
        data_feats = torch.load(args.precomputed_feats, map_location='cpu')
    else:
        hook = assign_storage_hook(model, args.layer, with_graph=True)
        data_feats, sample_cnt = load_good_features(
            model=model, loader=loader, how_many_batches=args.num_batches, 
            do_sample=True
        )
        hook.remove()
        saved_path = f'/scratch/inf0/user/mparcham/LargeFeats-{args.arch}-L{args.layer}-N{sample_cnt}.pth'
        assert not os.path.exists(saved_path)
        torch.save(data_feats.detach().cpu().float(), saved_path)
        print(f'Saved the features!\n Please run the code again with --precomputed_feats {saved_path}')
        exit()
        
    print(f'Training feature shapes: {data_feats.shape}')
    torch.cuda.empty_cache()

    save_path = f'Experiments/{args.dir_tree}/'
    os.makedirs(save_path, exist_ok=True)
    save_path += f'{args.config_name}'
    print(f"{save_path=}")
    solver = SAETrainer(
        n_components=args.nr_concepts, random_state=0,
        lr=args.lr, sparsity=args.sparsity, max_iter=args.max_iter,
        batch_size=args.batch_size,
    )

    solver.fit(data_feats[:, None], save_path=save_path,
        eval_model=model, eval_loader=val_loader, # For cross-val during training
        eval_layer=args.layer)
            
    print('Training Ended')

if __name__ == '__main__':
    args = get_args()
    train_imn(args)