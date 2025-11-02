import os
import math
import numpy as np
import torch
import argparse
import math
import json
import wandb

from os.path import join as ospj
from tqdm.rich import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from contextlib import nullcontext
from tqdm.rich import tqdm
from typing import Optional
from fact.training.saes import (
    log_stats, 
    MyTopKTrainer, MyBiasFreeTopKSAE, assert_no_bias
)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class TensorDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        return self.tensor[idx]
            
class SAETrainer(torch.nn.Module):
    def __init__(self, n_components, sparsity, lr, max_iter, 
                batch_size, random_state=0, verbose=True):
        super(SAETrainer, self).__init__()
        self.n_components = n_components # rank
        self.sparsity = sparsity 
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size # (individual feature vectors)
        self.random_state = random_state
        self.checkpoints = {i: None for i in range(max_iter)}

        self.verbose = verbose
        
    def fit(self, data_feats, save_path, eval_model=None, eval_loader=None, eval_layer=None):
        device = 'cuda'
        assert data_feats.ndim == 3, f'Expected (N, 1, E) but got {data_feats.shape}'
        self.n_features, self.feats_per_img, self.embd = data_feats.shape
        assert self.feats_per_img == 1
        print(f"{self.n_features} features of length {self.embd}")

        buffer = DataLoader(
            TensorDataset(data_feats), batch_size=self.batch_size,
            shuffle=True, num_workers=4, drop_last=False,
            collate_fn=lambda batch: torch.cat(batch, dim=0),
            persistent_workers=True
        )
        steps_per_epoch = len(buffer)
        n_steps = self.max_iter * steps_per_epoch
        print(f'{steps_per_epoch=} {next(iter(buffer)).shape=}')
        
        trainer_cfg = dict(
            # SAE config     
            trainer = MyTopKTrainer,  dict_class = MyBiasFreeTopKSAE,
            k = int(self.sparsity),
            auxk_alpha = 1/32,
            activation_dim = self.embd, dict_size = self.n_components,
            
            # LR scheduling
            lr = self.lr, steps=n_steps, warmup_steps=1*steps_per_epoch,
            decay_start=2*steps_per_epoch,
            
            device = device, seed = self.random_state,
            layer=0, lm_name='N/A', # These don't matter
            threshold_start_step = math.inf, # Disabled
        )
    
        train_sae(
            data=buffer, trainer_config=trainer_cfg,
            normalize_activations=True, # more of scaling than normalization
            steps=n_steps, device=device,
            
            # Checkpointing and logging
            save_steps=np.linspace(0, n_steps-1, 3, dtype='int'),
            save_dir=save_path,
            verbose=self.verbose, log_steps=(steps_per_epoch // 3),

            # Cross-validation throughout training
            eval_steps=2*steps_per_epoch,
            eval_model=eval_model, eval_loader=eval_loader, eval_layer=eval_layer,
        )

def train_sae(
    data, trainer_config: dict, steps: int,
    save_steps, save_dir, log_steps,
    normalize_activations:bool=False,
    verbose:bool=False,
    device:str="cuda", autocast_dtype: torch.dtype = torch.float32,
    eval_steps:Optional[int]=None, eval_loader=None,
    eval_model=None, eval_layer=None):
    """
    Adapted from trainSAE in dictionary_learning codebase
    https://github.com/saprmarks/dictionary_learning/blob/main/dictionary_learning/training.py
    """

    trainer_class = trainer_config.pop("trainer")
    trainer = trainer_class(**trainer_config)

    n_steps_per_epoch = len(data)
    n_feats_per_epoch = len(data.dataset)
    trainer.dead_feature_threshold = n_feats_per_epoch // 2
    print(f"{n_feats_per_epoch=} {trainer.dead_feature_threshold=}")
    
    print(f"Initialized the trainer and AE = {trainer.ae=}")
    
    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else \
        torch.autocast(device_type=device_type, dtype=autocast_dtype)

    # make save dirs, export config
    os.makedirs(save_dir, exist_ok=True)
    config = {"trainer": trainer.config}
    if hasattr(data, 'config'):
        config["buffer"] = data.config
    json.dump(config, open(ospj(save_dir, "config.json"), "w"), indent=4)

    if normalize_activations:
        from dictionary_learning.training import get_norm_factor
        norm_factor = get_norm_factor(data, steps=550)

        trainer.config["norm_factor"] = norm_factor
        trainer.ae.scale_biases(1.0)

    wandb.config.update(dict(trainer_config=trainer_config))
    wandb.config.update(dict(normalize_activations=normalize_activations))
    wandb.config.update(dict(norm_factor=norm_factor))
    
    if not os.path.exists(ospj(save_dir, "checkpoints")):
        os.mkdir(ospj(save_dir, "checkpoints"))
    
    step = 0; best_acc = -1
    while True:
        if step >= steps: break
        for act in tqdm(data): # One epoch
            step += 1
            if step >= steps: break
              
            log = None
            act = act.to(device_type, dtype=autocast_dtype) # Amin Diff
            if normalize_activations:
                act /= norm_factor

            if (step+1) % log_steps == 0:
                log = log_stats(
                    trainer, step, act, verbose=verbose,
                )
                wandb.log(log, step=step, commit=False)

            if (step+1) in save_steps:
                if log is None:
                    log = log_stats(
                        trainer, step, act, verbose=verbose,
                    )
                    wandb.log(log, step=step, commit=False)
                
                if normalize_activations:
                    trainer.ae.scale_biases(norm_factor)

                checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                checkpoint['log'] = log
                torch.save(checkpoint,
                    ospj(save_dir, "checkpoints", f"ae_{step}.pt"),
                )

                if normalize_activations:
                    trainer.ae.scale_biases(1 / norm_factor)
                print(f'Saved with {log=}')
            
            # Eval Accuracy
            if eval_steps is not None and (step+1) % eval_steps == 0:
                hook = assign_sae_hook(eval_model, eval_layer, trainer.ae, with_grad=False)
                correctness = eval_accuracy(eval_loader, eval_model)
                acc = correctness.float().mean()
                print(f"@ {step=} Val accuracy over {len(correctness)} Samples: {100*acc:0.2f}")
                wandb.log(dict(val_acc1=acc), step=step, commit=False)             
                hook.remove()
                
                if log is None:
                    log = log_stats(
                        trainer, step, act, verbose=verbose,
                    )
                    wandb.log(log, step=step, commit=False)
                log['val_acc'] = acc
                checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                checkpoint['log'] = log
                torch.save(checkpoint, ospj(save_dir, f"last.pt"))
                if acc >= best_acc:
                    torch.save(checkpoint, ospj(save_dir, f"best.pt"))
                    best_acc = acc
                    
                trainer.ae.zero_grad() # To be sure
                
            # Train for one step
            with autocast_context:
                trainer.update(step, act)

                # Log the activity of latents
                if (step+1) % n_steps_per_epoch == 0:
                    presence_th = int(0.6 * n_feats_per_epoch) # >= this is `too active'
                    absence_th = 0 # <= this is `dead'
                    too_active, fully_dead = trainer.activity_check(
                        activations=act, presence_th=presence_th, absence_th=absence_th
                    )
                    print(f'Activity: {too_active} highly active (>{presence_th}) and {fully_dead} dead latents')
                    wandb.log(dict(too_active=too_active, fully_dead=fully_dead), step=step, commit=False)
                
            wandb.log(dict(Lr=trainer.scheduler.get_last_lr()[0]), step=step, commit=True)
        
    # save final SAEs
    if normalize_activations:
        trainer.ae.scale_biases(norm_factor)
    
    final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
    torch.save(final, os.path.join(save_dir, "final.pt"))

    return trainer

def eval_sae(val_loader, model, sae_model,
        force_label, one_batch_only=False, custom_logit=None
    ): 
    all_contribs = []; all_presence = [];
    all_logits = []; all_correct_indices = []; all_l2_errors = [];
    for batch_idx, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        ret_dict = features_and_weights_sae(images, model, sae_model,
                labels=labels, force_label=force_label,
                with_graph=one_batch_only, custom_logit=custom_logit)
 
        logits = ret_dict.pop('logits')
        logits_to_explain = ret_dict.pop('logits_to_explain')
        orig_weights = ret_dict.pop('orig_weights')
        p_contribs = ret_dict.pop('p_contribs')
        p_presence = ret_dict.pop('p_presence')
        correct_indices = torch.arange( # Read and map the index for correctly classified samples
                    batch_idx*val_loader.batch_size, 
                    (batch_idx+1)*val_loader.batch_size
                )[ret_dict.pop('correct_indices')]
    
        if one_batch_only:
            return dict(
                images=ret_dict.pop('images'), orig_weights=orig_weights.cuda(),
                logits=logits.cuda(), logits_to_explain=logits_to_explain.cuda(), 
                labels=labels, p_presence=p_presence, p_contribs=p_contribs,
                correct_indices=correct_indices,
            )

        all_contribs += [p_contribs.detach().cpu()]
        all_presence += [p_presence.detach().cpu()]
        all_logits += [logits.detach().cpu()]
        all_correct_indices += [correct_indices]
        all_l2_errors += [ret_dict.pop('l2_errors')]

    all_contribs = torch.concat(all_contribs, dim=0)
    all_presence = torch.concat(all_presence, dim=0)
    all_logits = torch.concat(all_logits, dim=0)
    all_correct_indices = torch.concat(all_correct_indices, dim=0)
    all_l2_errors = torch.concat(all_l2_errors, dim=0)

    return all_contribs, all_logits, all_correct_indices, all_presence, all_l2_errors

def assign_sae_hook(model, lidx, sae_model, with_grad=True, hook_idx=None):
    def sae_hook_fn(module, input, output, storage, sae_model, with_grad):
        assert len(list(storage.keys())) == 0
        if with_grad:
            torch.set_grad_enabled(True);
        
        storage["intermediate_output"] = output
        if with_grad: storage["intermediate_output"].retain_grad()
        
        assert_no_bias(sae_model)
        
        if output.ndim == 3:
            is_vit = True; b, h, c = output.shape; w = 1;
            output_flat = output.flatten(0, 1)
        else:
            is_vit = False; b, c, h, w = output.shape
            output_flat = output.moveaxis(1, -1).flatten(0, 2)
        acts = sae_model.encode(output_flat)
        storage["intermediate_acts"] = acts
        if with_grad: storage["intermediate_acts"].retain_grad()

        output_hat = sae_model.decode(acts)
        if is_vit:
            storage["intermediate_rec"] = output_hat.reshape(b, h, c)
        else:
            storage["intermediate_rec"] = output_hat.reshape(b, h, w, c).moveaxis(-1, 1)
        if with_grad: storage["intermediate_rec"].retain_grad()
        output_hat = storage["intermediate_rec"]
        
        actual_err = output - output_hat.detach()
        storage["intermediate_err"] = actual_err
        if with_grad: storage["intermediate_err"].retain_grad()
        
        return output_hat
    
    model.hook_storage = {}
    model.hook_storage2 = {}
    model.hook_storage3 = {}
    for name, module in model.named_modules():
        if name == lidx:
            hook = module.register_forward_hook(
                lambda m, i, o: sae_hook_fn(m, i, o,
                    model.hook_storage if hook_idx is None else \
                    model.hook_storage2 if hook_idx == 2 else
                    model.hook_storage3 if hook_idx == 3 else None, sae_model, 
                    with_grad=with_grad)
            )
            return hook

def assign_channel_hook(model, lidx, with_grad=True):
    model.hook_storage = {}
    def channel_hook_fn(module, input, output, storage, with_grad):
        if with_grad: torch.enable_grad()
        assert len(list(storage.keys())) == 0
        storage["intermediate_output"] = output
        if with_grad: storage["intermediate_output"].retain_grad()

        if output.ndim == 3:
            is_vit = True; b, h, c = output.shape; w = 1;
            output_flat = output.flatten(0, 1)
        else:
            is_vit = False; b, c, h, w = output.shape
            output_flat = output.moveaxis(1, -1).flatten(0, 2)
        storage["intermediate_acts"] = output_flat # Fake Acts
        if with_grad: storage["intermediate_acts"].retain_grad()

        if is_vit:
            storage["intermediate_rec"] = output_flat.reshape(b, h, c)
        else:
            storage["intermediate_rec"] = output_flat.reshape(b, h, w, c).moveaxis(-1, 1)
        if with_grad: storage["intermediate_rec"].retain_grad()
        output = storage["intermediate_rec"]
        
        storage["intermediate_err"] = torch.zeros_like(output).cuda()
        storage["intermediate_err"].requires_grad_()
        if with_grad: storage["intermediate_err"].retain_grad()
         
        return output
    
    for name, module in model.named_modules():
        if name == lidx:
            hook = module.register_forward_hook(
                lambda m, i, o: channel_hook_fn(m, i, o,
                    model.hook_storage, with_grad=with_grad)
            )
            return hook

def assign_storage_hook(model, lidx, with_graph=True):
    model.hook_storage = {}
    def hook_fn(module, input, output, storage, with_graph):
        if with_graph:
            torch.set_grad_enabled(True)
            output.requires_grad_()
        assert 'intermediate_output' not in storage
        storage["intermediate_output"] = output
        if with_graph:
            storage["intermediate_output"].retain_grad()
        return storage["intermediate_output"]
    
    for name, module in model.named_modules():
        if name == lidx:
            hook = module.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, model.hook_storage, with_graph))
            print(f"Hook registered to {name}")
            return hook

def just_features(images, model, labels):
    b_size = 64; n_batches = 1 + len(images) // b_size

    all_logits = []; all_logits_to_explain = [];
    all_rec_logits = []; all_p_contribs = [];
    all_indices = []; all_p_presence = []
    all_l2_errors = [];
    for i in range(n_batches):
        img = images[b_size * i: b_size * (i+1)].cuda()
        img.requires_grad = True
        lbl = labels[b_size * i: b_size * (i+1)]

        logits = model(img)

        if hasattr(model, 'logit_layer'):
            logits -= model.logit_layer.logit_bias
        if logits.ndim > 2:
            logits = F.adaptive_avg_pool2d(logits, (1, 1)).flatten(-3)
        preds = logits.argmax(dim=-1).detach().cpu()
        
        pre_sae_features = model.hook_storage['intermediate_output']
        if pre_sae_features.ndim == 3:
            is_vit = True; b, h, c = pre_sae_features.shape; w = 1;
        else:
            is_vit = False; b, c, h, w = pre_sae_features.shape
        sae_acts = model.hook_storage['intermediate_acts']
        sae_err = model.hook_storage['intermediate_err']

        p_presence = sae_acts.reshape(b, h, w, sae_acts.shape[1]).sum((1, 2))
        all_p_presence += [p_presence]
        correct_indices = torch.where(preds == lbl)[0].detach().cpu()
        all_indices += [correct_indices]
        all_logits += [logits.detach()]

        # These are just filled for compatibility with other functions!
        logits_to_explain = logits[range(len(logits)), preds]
        all_logits_to_explain += [logits_to_explain.detach()]
        all_rec_logits += [torch.zeros_like(logits_to_explain)]
        p_contribs = torch.zeros_like(p_presence)
        all_p_contribs += [p_contribs.detach()]
        all_l2_errors += [sae_err.detach().pow(2).sum(dim=-1 if is_vit else 1).sqrt()]
        
        img.grad = sae_err = sae_acts = None;
        torch.cuda.empty_cache()
        model.hook_storage = {}
            
    return dict(
        images=None, orig_weights=torch.Tensor([]),
        p_contribs=torch.concat(all_p_contribs, dim=0),
        p_presence=torch.concat(all_p_presence, dim=0),
        logits=torch.concat(all_logits, dim=0),
        logits_to_explain=torch.concat(all_logits_to_explain, dim=0),
        rec_logits=torch.concat(all_rec_logits, dim=0),
        correct_indices=torch.concat(all_indices, dim=0),
        l2_errors=torch.concat(all_l2_errors, dim=0),
    )
  
def features_and_weights_sae(
    images, model, sae_model, labels, with_graph=False,
    force_label=False, custom_logit=None):
    if not hasattr(model, 'explanation_mode'):
        assert sae_model is None and not (with_graph or force_label)
        return just_features(images, model, labels)

    b_size = 64; # Adjust this if you get OOM
    n_batches = 1 + len(images) // b_size
    assert (not with_graph) or n_batches == 1,\
            f'This is only supported for one batch of <{b_size} at a time!'

    all_logits = []; all_logits_to_explain = [];
    all_p_contribs = [];
    all_orig_weights = []; all_indices = []; all_p_presence = []
    all_l2_errors = [];

    with model.explanation_mode():
        for i in range(n_batches):
            img = images[b_size * i: b_size * (i+1)].cuda()
            img.requires_grad = True
            lbl = labels[b_size * i: b_size * (i+1)]

            logits = model(img)

            if hasattr(model, 'logit_layer'):
                logits -= model.logit_layer.logit_bias
            if logits.ndim > 2:
                logits = F.adaptive_avg_pool2d(logits, (1, 1)).flatten(-3)
            preds = logits.argmax(dim=-1).detach().cpu()
            
            pre_sae_features = model.hook_storage['intermediate_output']
            if pre_sae_features.ndim == 3:
                is_vit = True; b, h, c = pre_sae_features.shape; w = 1;
            else:
                is_vit = False; b, c, h, w = pre_sae_features.shape
            sae_acts = model.hook_storage['intermediate_acts']
            sae_rec = model.hook_storage['intermediate_rec']
            sae_err = model.hook_storage['intermediate_err']

            if force_label:
                logits_to_explain = logits[range(len(logits)), lbl]
            elif custom_logit is not None:
                logits_to_explain = logits[range(len(logits)), custom_logit]
            else:
                logits_to_explain = logits[range(len(logits)), preds]
            
            logits_to_explain.sum(dim=0).backward(
                inputs=[
                    sae_acts, sae_rec, img
                ] if with_graph else [
                    sae_acts, sae_rec,
                ],
                retain_graph=with_graph, create_graph=False
            )
            rec_logits = (sae_rec.grad.detach() * sae_rec).flatten(1).sum(1)        
            p_contribs = (sae_acts.grad.detach() * sae_acts).reshape(b, h, w, sae_acts.shape[1])
            p_contribs = p_contribs.sum((1, 2))
            p_presence = sae_acts.reshape(b, h, w, sae_acts.shape[1]).sum((1, 2))

            sae_acts.grad = sae_rec.grad = sae_err.grad = None

            # Assert the completeness of the model
            if not (is_vit or torch.allclose(
                rec_logits.detach(), logits_to_explain.detach(), 1e-1)):
                print(f"Warning!!! High Error at Logit Reconstruction {(rec_logits.detach()- logits_to_explain.detach()).max().item()}")

            # Assert the completeness of the SAE
            if sae_model is not None:
                assert assert_no_bias(sae_model) and torch.allclose(
                    p_contribs.detach().sum(1), rec_logits.detach(), 1e-2)

            all_logits += [logits.detach()]
            all_logits_to_explain += [logits_to_explain.detach()]
            all_p_contribs += [p_contribs.detach()]
            all_p_presence += [p_presence]
            all_l2_errors += [sae_err.detach().pow(2).sum(dim=-1 if is_vit else 1).sqrt()]
            sae_acts = sae_rec = None
            if with_graph:
                all_orig_weights += [img.grad.detach()];
            
            img.grad = sae_err = None;
            torch.cuda.empty_cache()
            correct_indices = torch.where(preds == lbl)[0].detach().cpu()
            all_indices += [correct_indices]
            model.hook_storage = {}
            
    return dict(
        images=img if with_graph else None,
        orig_weights=torch.concat(all_orig_weights, dim=0) if with_graph else torch.Tensor([]),
        p_contribs=p_contribs if with_graph else torch.concat(all_p_contribs, dim=0),
        p_presence=torch.concat(all_p_presence, dim=0),
        logits=torch.concat(all_logits, dim=0),
        logits_to_explain=torch.concat(all_logits_to_explain, dim=0),
        correct_indices=torch.concat(all_indices, dim=0),
        l2_errors=torch.concat(all_l2_errors, dim=0),
    )

def eval_accuracy(val_loader, model): 
    print('Evaluating Accuracy')
    with torch.no_grad():
        correctness = []
        for batch_idx, (images, labels) in enumerate(val_loader):
            output = model(images.cuda())
            preds = output.argmax(dim=1).cpu()
            correctness += [preds == labels]
            model.hook_storage = {}
            model.hook_storage2 = {}
            model.hook_storage3 = {}
    return torch.cat(correctness, dim=0)

def load_good_features(model, loader, how_many_batches, do_sample=True):
    max_batch = min(how_many_batches, len(loader))
    vlist = []; sample_cnt = 0; correct_cnt = 0
    with model.explanation_mode():
        for batch_idx, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            if batch_idx >= max_batch: 
                print(f'Ending after {max_batch} / {len(loader)} batches')
                break
            if batch_idx % 1000 == 0: 
                print(f'Reached {batch_idx=}')
            
            torch.set_grad_enabled(False)
            logits = model(images.cuda())
            preds = logits.argmax(dim=1).detach().cpu()
            inp = model.hook_storage['intermediate_output']
            model.hook_storage = {}
            if inp.ndim == 3:
                is_vit = True; b, h, c = inp.shape; w = 1;
                embeddings = inp.detach()
            else:
                is_vit = False; b, c, h, w = inp.shape
                embeddings = inp.flatten(2).moveaxis(-1, 1).detach()
            
            
            ##### Sample the features based on their importance to the output #####
            # Adapt the nr of sampled feature vectors based on vector size
            # Makes stored features (of all layers and architecture) to take same disk space
            # Currently configured as 72 features of length 512 per image --> 170G for whole ImageNet
            # Simply reduce the 72 if you want smaller dataset (e.g. 36 would lead to ~85G)
            n_samples = int((72 * 512) / c) if do_sample else h*w

            inp.grad = None
            logits[range(len(images)), preds].sum(dim=0).backward(
                inputs = [inp], retain_graph=False, create_graph=False
            )
            contribs = (inp.grad.detach() * inp.detach())
            if is_vit:
                contribs = contribs.sum(2).abs() # B, HW
            else:
                contribs = contribs.sum(1).flatten(1).abs() # B, HW
            inp.grad = None;
            sampled_indices = torch.multinomial(
                contribs, num_samples=min(n_samples, h*w),
                replacement=False
            )
            sampled_indices = sampled_indices[..., None].expand(-1, -1, c)
            embeddings = torch.gather(embeddings, dim=1, index=sampled_indices).contiguous()
            
            if batch_idx == 0: 
                print(f"Orig: {inp.shape=}  | {n_samples=} of {h*w} | Modified {embeddings.shape=}")
            embeddings = embeddings.flatten(0, 1).detach().cpu().float()
            vlist += [embeddings]
            
            sample_cnt += b
            correct_cnt += len(images)
            inp = logits = contribs = embeddings = None

            
        vlist = torch.concat(vlist, dim=0)
            
    print(f"&&&&& Final {sample_cnt=} {vlist.shape=} {correct_cnt=} &&&&&")
    return vlist, sample_cnt

def get_args():
    parser = argparse.ArgumentParser()
    
    # Experiment Configs (training and evaluation)
    parser.add_argument("--layer", type=str, help="Layer Name")
    parser.add_argument("--arch", type=str, help="Architecture")
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--method', choices=['BiasFreeTopK', 'channels', 'craft', 'crp'], type=str, default=None)
    parser.add_argument("--nr_concepts",type=int, help="Total number of concepts")
    parser.add_argument('--sparsity', type=int, default=0, help='K-sparisty factor for TOPK-SAE; Set 0 for channels')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate for SAE training')

    parser.add_argument('--quantile_per_concept', type=float, default=None, 
                        help='Quantile to consider to evaluation')
    
    # Training arguments
    parser.add_argument('--num_batches', type=int, default=5000,
                        help='Maximum number of batches to export features from (before SAE training)')
    parser.add_argument('--precomputed_feats', type=str, default=None,
                        help="Path to the stored features for SAE training")
    parser.add_argument("--max_iter", type=int, help="Number of epochs for SAE training")
    parser.add_argument('--batch_size', type=int, help="Batchsize for SAE training")
    
    parser.add_argument('--bcos_layers', action='store_true', help="Whether working with B-cos layers")
    parser.add_argument('--download', default=None, choices=['v0.1'], # Please check github for new releases
                        help="Whether to download checkpoints from github releases. \
                        None (default) means loading your own checkpoints")
    
    args = parser.parse_args()
    assert args.bcos_layers ^ args.arch.startswith('std_'),\
        'It should always be one of the two'
    
    # These will be used for creating paths and sub-dirs when doing analysis
    args.dir_tree = f"{args.arch}/{args.layer}/{args.exp_name}"
    args.config_name = f"K{args.nr_concepts}-S{args.sparsity}-LR{args.lr}"
    return args