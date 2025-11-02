import numpy as np
import torch
from torch import nn
from typing import Optional, Callable
from collections import namedtuple
from functools import partial
from dictionary_learning.trainers.top_k import (
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
    TopKTrainer, AutoEncoderTopK
)

def assert_no_bias(sae_model):
    if hasattr(sae_model, 'bias'):
        assert torch.all(sae_model.bias == 0), 'non zero biases!'
    if hasattr(sae_model, 'b_dec'):
        assert torch.all(sae_model.b_dec == 0), 'non zero biases!'
    assert sae_model.encoder.bias is None or not sae_model.encoder.bias, 'non zero biases!'
    assert sae_model.decoder.bias is None, 'non zero biases!'
    return True
class MyBiasFreeTopKSAE(AutoEncoderTopK):
    def __init__(self, *args,  **kwargs):
        super(MyBiasFreeTopKSAE, self).__init__(*args, **kwargs)
        del self.encoder # Amin Diff: encoder no bias
        self.encoder = nn.Linear(self.activation_dim, self.dict_size, bias=False)
        self.encoder.weight.data = self.decoder.weight.T.clone()

        del self.b_dec # Amin Diff: No Bias!
        self.b_dec = torch.zeros(self.activation_dim, requires_grad=False).cuda()
        self.has_no_bias = True
        self.feat_mean = None

    def scale_biases(self, scale: float):
        assert_no_bias(self)
        
    def encode(self, x: torch.Tensor, return_topk: bool = False, use_threshold: bool = False):
        assert not use_threshold and torch.all(self.b_dec==0) # Amin Diff
        if self.feat_mean is not None:
            raise NotImplementedError
            # x -= self.feat_mean
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            raise NotImplementedError # Amin Diff

        if self.feats_per_img is not None and self.feats_per_img > 1:
            raise NotImplementedError
        else:
            post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

            # We can't split immediately due to nnsight
            tops_acts_BK = post_topk.values
            top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x):
        ret = super().decode(x)
        if self.feat_mean is not None:
            raise NotImplementedError
            # ret += self.feat_mean
        return ret

class MyTopKTrainer(TopKTrainer):
    def __init__(self, *args, **kwargs):
        super(MyTopKTrainer, self).__init__(*args, **kwargs)
        self.dead_feature_threshold = 20_000_000 # (will be overwritten)
        self.top_k_aux=256
        self.ae.feats_per_img = 1 # Deprecated. Assume every feature patch to be an "image"
        self.num_present_images = torch.zeros(self.ae.dict_size, dtype=torch.int64, device='cuda')
        
        lr_fn = my_lr_schedule(
            total_steps=self.steps, warmup_steps=self.warmup_steps, decay_start=self.decay_start,
            resample_steps=None, sparsity_warmup_steps=None
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
        
    def update(self, step, x): # Amin Diff: everything like parent except step=0 behaviour
        if step == 0: # Amin Diff
            assert self.ae.has_no_bias
            print('Warning! At step 0 skipped setting the b_dec')

        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()
    
    def activity_check(self, activations, presence_th, absence_th):
        too_active = self.num_present_images >= presence_th
        fully_dead = self.num_present_images <= absence_th
        self.num_present_images *= 0
        return too_active.float().sum().item(), fully_dead.float().sum().item()
    
    def loss(self, x, step=None, logging=False): # Amin Diff: Same as parent except activity check
        # Run the SAE
        f, top_acts_BK, top_indices_BK, post_relu_acts_BF = self.ae.encode(
            x, return_topk=True, use_threshold=False
        )

        if step > self.threshold_start_step:
            raise NotImplementedError # Amin Diff
            self.update_threshold(top_acts_BK)

        x_hat = self.ae.decode(f)

        # Measure goodness of reconstruction
        e = x - x_hat

        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts_BK.size(1)

        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[top_indices_BK.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = (
            self.get_auxiliary_loss(e.detach(), post_relu_acts_BF) if self.auxk_alpha > 0 else 0
        )

        loss = l2_loss + self.auxk_alpha * auxk_loss


        if not logging:
            # Amin Diff
            assert self.ae.feats_per_img is not None 
            _B, _K = top_indices_BK.shape
            top_indices_IK = top_indices_BK.view(
                    _B // self.ae.feats_per_img, self.ae.feats_per_img, _K
                ).flatten(1, 2).unique(dim=1)
            for fired_indices in top_indices_IK:
                self.num_present_images[fired_indices] += 1
            
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {"l2_loss": l2_loss.item(), "auxk_loss": auxk_loss.item(), "loss": loss.item()},
            )

def my_lr_schedule(
    total_steps: int,
    warmup_steps: int,
    decay_start: Optional[int] = None,
    resample_steps: Optional[int] = None,
    sparsity_warmup_steps: Optional[int] = None) -> Callable[[int], float]:
    """
    Similar to original get_lr_schedule of , except that decay_start and resampling are not mutualy exclusive.
    The decay is also cosine decay instead of linear.
    """
    print('Warning!!! Using custom lr scheduler!')
    if decay_start is not None:
        # assert resample_steps is None
        assert 0 <= decay_start < total_steps, "decay_start must be >= 0 and < steps."
        assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
        if sparsity_warmup_steps is not None:
            assert decay_start > sparsity_warmup_steps, (
                "decay_start must be > sparsity_warmup_steps."
            )

    assert 0 <= warmup_steps < total_steps, "warmup_steps must be >= 0 and < steps."

    def lr_schedule(step: int, warmup_steps: int, decay_start: int, resample_steps: int) -> float:
        if step < warmup_steps:
            # Warm-up phase
            return step / warmup_steps

        if decay_start is not None and step >= decay_start:
            left, total = (total_steps - step), (total_steps - decay_start)
            t = 1 - (left/total)
            # Assuming Base --> 0
            return 0.5 * (1 + np.cos(np.pi * t))
        
        # Only consider resampling if you're not already decaying!
        if resample_steps is not None:
            assert 0 < warmup_steps < resample_steps < total_steps,\
                "resample_steps must be > 0 and < steps."
            return min((step % resample_steps) / warmup_steps, 1.0)

        # Constant phase
        return 1.0
    
    lr_schedule = partial(lr_schedule, 
        warmup_steps=warmup_steps, decay_start=decay_start, resample_steps=resample_steps)
    
    return lr_schedule

def log_stats(trainer, step: int, act: torch.Tensor,
    verbose: bool=False, feats_per_img: int=None):
    """
    Adapted from dictionary_learning codebase 
    https://github.com/saprmarks/dictionary_learning/blob/main/dictionary_learning/training.py
    """
    with torch.no_grad():
        # quick hack to make sure all trainers get the same x
        log = {}
        act = act.clone()

        act, act_hat, f, losslog = trainer.loss(act, step=step, logging=True)

        # Amin Diff: measure image-L0 (only possible if you don't shuffle all training set patch-wise)
        # Our code currently doesn't use this
        if feats_per_img is not None and feats_per_img > 1:
            n_features, latent_size = f.shape
            n_images = n_features // feats_per_img
            f_instancewise = f.reshape(n_images, feats_per_img, latent_size)
            log['l0_img'] = (f_instancewise.abs().sum(1) != 0).float().sum(dim=-1).mean(dim=0).item()
        else:
            log['l0_img'] = 0
            
        # L0
        l0 = (f != 0).float().sum(dim=-1).mean().item()
        # fraction of variance explained
        total_variance = torch.var(act, dim=0).sum()
        residual_variance = torch.var(act - act_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
        log[f"frac_variance_explained"] = frac_variance_explained.item()


        if verbose:
            print(f"Step {step}: L0= {l0:0.3f} L0_img={log['l0_img']:0.3f}, frac_variance_explained = {frac_variance_explained:0.5f}  "+\
                f"Rec={losslog['l2_loss']:0.5f}") # Amin Diff

        # log parameters from training
        log.update({f"{k}": v.cpu().item() if isinstance(v, torch.Tensor) else v for k, v in losslog.items()})
        log[f"l0"] = l0
        trainer_log = trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            log[f"{name}"] = value

        return log

def load_SAE(args):
    if args.download is not None:
        repo = 'm-parchami/FaCT'; release_tag = args.download; 
        ckpt_name = f'{release_tag}-{args.arch}-{args.layer}-{args.method}-{args.config_name}.pt'
        ckpt_path = f'https://github.com/{repo}/releases/download/{release_tag}/{ckpt_name}'
        print(f"Fetching checkpoint from {ckpt_path} via torch.hub")
        checkpoint = torch.hub.load_state_dict_from_url(
            url=ckpt_path, progress=True, check_hash=False,
            map_location='cpu',
        )
    else:
        ckpt_path = f'Experiments/{args.dir_tree}/{args.config_name}.pt' 
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    train_log = checkpoint.pop('log', {})
    
    n_components, act_dim  = checkpoint['encoder.weight'].shape
    assert n_components == args.nr_concepts, 'SAE config doesn\'t match the weight dimensions!'
    assert 'encoder.bias' not in checkpoint \
        and 'decoder.bias' not in checkpoint, 'The SAE checkpoints must be bias-free!'
    
    sae_model = MyBiasFreeTopKSAE(activation_dim=act_dim, dict_size=n_components, k=args.sparsity)
    sae_model.feats_per_img = None
        
    sae_model.load_state_dict(checkpoint)
    sae_model.cuda()

    return sae_model, ckpt_path, train_log