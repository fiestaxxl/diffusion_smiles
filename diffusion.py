import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import decode
from tqdm import tqdm

import math
import numpy as np

import random


class GaussianDiffusion():
    def __init__(self, timesteps, vocab, weights=None, predict_xstart = True, rescale_timesteps=False):
        self.timesteps = timesteps
        self.vocab = vocab
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        self.weights = weights
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart

    def initialize(self, init_type = 'cosine'):
        if init_type == 'cosine':
            self.betas = self.cosine_beta_schedule(timesteps=self.timesteps)
        elif init_type == 'linear':
            self.betas = self.linear_beta_schedule(timesteps=self.timesteps)
        elif init_type == 'quadratic':
            self.betas = self.quadratic_beta_schedule(timesteps=self.timesteps)
        elif init_type == 'sigmoid':
            self.betas = self.sigmoid_beta_schedule(timesteps=self.timesteps)
        else:
            raise NotImplementedError()
        
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod_next = F.pad(self.alphas_cumprod[1:], (0, 1), value=0.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            F.pad(self.posterior_variance[:-1], (1,0), value=self.posterior_variance[1])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def quadratic_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    def sigmoid_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = self.extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self.extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance
    
    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        #x_t = sqrt(alphas_cumprod)*x_0 + N(0, (1-alphas_cumprod)I)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        if mask == None:
            return x_noisy
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask==0, x_start, x_noisy)
 
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
      
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)
        # print(x.shape)
        model_output = model(x, self._scale_timesteps(t))
        
        # for fixedlarge, we set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        model_variance = F.pad(self.betas[:-1], (1,0), value=self.posterior_variance[1])
        model_log_variance = torch.log(F.pad(self.betas[:-1], (1,0), value=self.posterior_variance[1]))

        
        model_variance = self.extract(model_variance, t, x.shape)
        model_log_variance = self.extract(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                # print(denoised_fn)
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            pred_xstart = process_xstart(model_output)
        else:
            ### model is used to predict eps
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )
        #print(model_mean.shape, model_log_variance.shape, pred_xstart.shape, x.shape)
        assert (
            model_mean.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    @torch.no_grad()
    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None, mask=None, x_start=None,):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = torch.randn_like(x)
            replace_mask = torch.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = torch.randn_like(noise[replace_mask])
                replace_mask = torch.abs(noise) > top_p
            assert (torch.abs(noise) <= top_p).all()

        else:
            noise = torch.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        if mask == None:
            pass
        else:
            sample = torch.where(mask==0, x_start, sample)

        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"], 
            "out": out
        }
    
    @torch.no_grad()
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        time_steps = 200,
        noise=None,
        clip_denoised=True,
        model_kwargs=None,
        device=None,
        progress=False,
        denoised_fn_cur=None,
        top_p=None,
        mask=None,
        x_start=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None: # custom your the start point of x_0
            sample_x = noise
        else:
            sample_x = torch.randn(*shape, device=device)
        
        if time_steps is not None:
            indices = list(range(time_steps))[::-1]
        else:
            indices = list(range(self.timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices: # from T to 0
            t = torch.tensor([i] * shape[0], device=device).long()
            out = self.p_sample(
                model,
                sample_x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn_cur,
                model_kwargs=model_kwargs,
                top_p=top_p,
                mask=mask,
                x_start=x_start
            )
            yield out
            sample_x = out["sample"]

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            device=device,
        ):
            final.append(sample['sample'])
        return final
    
    def _get_x_start(self, x_start_mean, std):
        '''
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        '''
        noise = torch.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return (
             x_start_mean + std * noise
        )

    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else: # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, lambda_validity=0.15):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)  # bsz, seqlen, vocab

        # tokens = logits.softmax(dim=-1).argmax(dim=-1) #bsz, seq_len
        # validity_mask = self.check_smiles_validity(tokens, self.reverse_vocab)  # Returns 1 for valid, 0 for invalid
        # validity_penalty = (1 - validity_mask.float()).mean()  # Penalize invalid generations

        # print(logits.shape)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), #[seq_len*bs, vocab]
                               input_ids.view(-1)).view(input_ids.shape) #[bsz, seq_len]
       
        if mask != None:
            decoder_nll *= mask
  
        if mask != None:
            decoder_nll = decoder_nll.sum(dim=-1)/mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1) #[bsz]

        return decoder_nll# + lambda_validity * validity_penalty

    @staticmethod
    def check_smiles_validity(token_ids, itos):
        """
        Check SMILES validity by reconstructing and validating syntax.
        :param token_ids: batch of generated SMILES sequences (as token IDs).
        :return: Tensor of shape (batch_size,) with 1 for valid SMILES and 0 for invalid.
        """
        from rdkit import Chem
        from rdkit.rdBase import BlockLogs

        def validate(s):
            block = BlockLogs()
            return Chem.MolFromSmiles(s)
        
        smiles_strings = [decode(x.cpu().tolist(), itos) for x in token_ids] #bs

        validity = [1 if validate(s) else 0 for s in smiles_strings]
        return torch.tensor(validity, dtype=torch.float, device=token_ids.device)
    
    @staticmethod
    def create_attention_mask(token_ids, vocab, boost_factor):
        """
        Create an attention mask for protecting parentheses and ring numbers in SMILES sequences.
        
        :param token_ids: Tensor of tokenized SMILES sequences (batch_size, seq_len).
        :param vocab: Dictionary mapping tokens to indices.
        :return: Binary mask of shape (batch_size, seq_len, seq_len) where 0 means "prevent attention to this position".
        """
        batch_size, seq_len = token_ids.shape
        #mask = torch.ones((batch_size, seq_len, seq_len), device=token_ids.device)
        bias = torch.zeros((batch_size, seq_len, seq_len), device=token_ids.device)

        
        # Define protected tokens (parentheses and ring numbers 1-9)
        protected_tokens = [vocab[tok] for tok in ['(', ')'] if tok in vocab]
        
        for token in protected_tokens:
            protected_positions = (token_ids == token).unsqueeze(1).expand(-1, seq_len, -1)
            #mask = mask.masked_fill(protected_positions, 0)  # Prevent attention to protected tokens
            bias += protected_positions.float() * boost_factor 
        
        # return mask
        return bias

    def bracket_depth_loss(self, x_t, get_logits, input_ids):
        pred_ids = get_logits(x_t).argmax(dim=-1)  # (batch, seq_len)

        open_mask_pred = (pred_ids == self.vocab['(']).int()
        close_mask_pred = (pred_ids == self.vocab[')']).int()

        open_mask_tgt = (input_ids == self.vocab['(']).int()
        close_mask_tgt = (input_ids == self.vocab[')']).int()

        def compute_depths(open_mask, close_mask):
            # running depth: depth[i] = open[:i+1].sum() - close[:i+1].sum()
            # Use cumulative sum for fast computation
            depth = torch.cumsum(open_mask - close_mask, dim=1)
            # Clamp to avoid negative depths
            return depth.clamp(min=0).float()

        pred_depths = compute_depths(open_mask_pred, close_mask_pred)
        tgt_depths = compute_depths(open_mask_tgt, close_mask_tgt)

        # Compute MSE loss per sample, average over tokens, then over batch
        loss = torch.nn.functional.mse_loss(pred_depths, tgt_depths)
        #loss = loss.mean(dim=1)  # average per sample
        return loss  # shape: (batch,)

    def training_losses_seq2seq(self, model, t, input_ids, corrupted_input_ids=None, tau=None, noise=None, boost_factor=0.0):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. # not used unless fixing the input embeddings
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        input_ids_x = input_ids.to(t.device) #[bs, seq_len]
        if corrupted_input_ids is not None:
            corrupted_input_ids_x = corrupted_input_ids.to(t.device)


        x_start_mean = model.get_embeds(input_ids_x) #[bs, seq_len, emb_dim]
        if corrupted_input_ids is not None:
            x_start_mean_corrupted = model.get_embeds(corrupted_input_ids_x)
        
        std = self.extract(self.sqrt_one_minus_alphas_cumprod,
                                   torch.full((x_start_mean.shape[0],), 0, device=x_start_mean.device, dtype=torch.long),
                                   x_start_mean.shape)

        x_start = self._get_x_start(x_start_mean, std)
        if corrupted_input_ids is not None:
            x_start_corrupted = self._get_x_start(x_start_mean_corrupted, std)

        if noise is None:
            noise = torch.randn_like(x_start)

        if corrupted_input_ids is None:
            x_t = self.q_sample(x_start, t, noise=noise, mask=None) # reparametrization trick.
        else:
            corrupt_mask = (t>tau).view(-1, 1, 1)
            #print(corrupt_mask.shape, x_start.shape, x_start_corrupted.shape)
            x_start = torch.where(corrupt_mask, x_start, x_start_corrupted)
            x_t = self.q_sample(x_start, t, noise=noise, mask=None) # reparametrization trick.

        get_logits = model.get_logits
        terms = {}
        

        #attention_mask = self.create_attention_mask(input_ids, self.vocab, boost_factor).to(t.device)
        attention_mask = None
        target = x_start
        model_output = model(x_t, self._scale_timesteps(t), mask=attention_mask)

        terms['mse'] = F.mse_loss(target, model_output, reduction='none').mean(dim=(1, 2))
        #terms["mse"] = mean_flat((target - model_output) ** 2)

        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart'] # predicted_xstart = model_output
        t0_mask = (t == 0)
        #t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
        t0_loss = F.mse_loss(x_start_mean, model_out_x_start, reduction='none').mean(dim=(1,2))
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        out_mean, _, _ = self.q_mean_variance(x_start, torch.full((x_start_mean.shape[0],), self.timesteps - 1, device=x_start_mean.device, dtype=torch.long))
        tT_loss =  torch.sqrt(mean_flat(out_mean ** 2))

        if corrupted_input_ids is None:
            decoder_nll = self._token_discrete_loss(x_start, get_logits, input_ids_x) # embedding regularization
            terms["nll"] = self._token_discrete_loss(model_out_x_start, get_logits, input_ids_x, mask=None) # x_0->model_out_x_start

            bracket_loss = self.bracket_depth_loss(x_t, get_logits, input_ids_x)
            terms["loss"] = terms["mse"] + decoder_nll + tT_loss + 0.3*bracket_loss
        else:
            terms['loss'] = terms['mse']

        return terms

    def training_losses(self, model, *args, **kwargs):
        return self.training_losses_seq2seq(model, *args, **kwargs)
    




def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # print(logvar2.shape)
    # temp1 = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2))
    # print(f'const = {temp1.mean()}, coef={(torch.exp(-logvar2) * 0.5).mean()}, mse={((mean1 - mean2) ** 2).mean().item()}')

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def gaussian_density(x, *, means, log_scales):
    from torch.distributions import Normal
    normal_dist = Normal(means, log_scales.exp())
    logp = normal_dist.log_prob(x)
    return logp 


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

