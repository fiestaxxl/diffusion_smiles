import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

import functools
from torch.optim import AdamW


from abc import ABC, abstractmethod

import logging

from utils import decode, process_decode
from validate import validity


# Create a separate logger for tools.py
logger = logging.getLogger("train_logger")
logger.setLevel(logging.INFO)

# Configure a file handler for the tools logger
file_handler = logging.FileHandler("train.log", mode='w')
file_handler.setLevel(logging.INFO)

# Set a formatter for the tools logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the tools logger
logger.addHandler(file_handler)


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len, vocab, corrupt=False, corrupt_ratio=0.1):
        self.smiles_list = smiles_list
        self.encode = tokenizer
        self.max_len = max_len
        self.stoi = vocab
        self.corrupt = corrupt
        self.corrupt_ratio = corrupt_ratio

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        pure_tokens = self.encode(self.smiles_list[idx], self.stoi)

        tokens_with_symbols = [self.stoi['<sos>']] + pure_tokens + [self.stoi['<eos>']]
        pad_len = self.max_len - len(tokens_with_symbols)
        tokens_with_symbols = tokens_with_symbols + [self.stoi['<pad>']] * pad_len # Padding

        if self.corrupt:
            corrupt_tokens = np.array([self.stoi[str(i)] for i in range(1, 10)] + [self.stoi['('], self.stoi[')']])
            tokens = list(self.corrupt_tokens(pure_tokens, corrupt_tokens, self.corrupt_ratio))
            assert len(tokens) == len(pure_tokens), f"{len(tokens)}, {len(pure_tokens)}"

            corrupted_tokens_with_symbols = [self.stoi['<sos>']] + tokens + [self.stoi['<eos>']]
            corrupted_tokens_with_symbols = corrupted_tokens_with_symbols + [self.stoi['<pad>']] * pad_len # Padding
            corrupted_tokens_with_symbols = torch.tensor(corrupted_tokens_with_symbols, dtype=torch.long)
        else:
            corrupted_tokens_with_symbols = torch.tensor([0 for _ in range(len(tokens_with_symbols))])
        assert len(tokens_with_symbols) == len(corrupted_tokens_with_symbols), f'{len(tokens_with_symbols)}, {len(corrupted_tokens_with_symbols)}'
        return torch.tensor(tokens_with_symbols, dtype=torch.long), pad_len, corrupted_tokens_with_symbols

    def corrupt_tokens(self, token_ids, corrupt_values, max_corruption_ratio=0.1):
        """
        Corrupts a sequence of token IDs by randomly inserting tokens from {1-9, 78, 79}.
        Ensures at least one corruption but no more than 10% of the total tokens.
        
        Args:
            token_ids (np.ndarray): 1D array of token IDs.
            corrupt_values (list | np.ndarray): 1D array of corrupt tokens IDs
            max_corruption_ratio (float): Max fraction of tokens to corrupt (default 10%).
        
        Returns:
            np.ndarray: Corrupted token sequence.
        """
        token_ids = np.array(token_ids)  # Ensure it's a NumPy array
        seq_len = len(token_ids)

        # Determine the number of corruptions (at least 1, at most 10% of seq_len)
        num_corruptions = int(max(1, min(int(seq_len * max_corruption_ratio), seq_len)))  # Ensure at least 1 corruption

        # Select `num_corruptions` unique positions to corrupt
        corruption_indices = np.random.choice(seq_len, num_corruptions, replace=False)

        # Possible corruption values
        corrupt_values = np.array(corrupt_values)

        # Generate random corrupt tokens
        random_tokens = np.random.choice(corrupt_values, num_corruptions)

        # Create a corrupted copy
        corrupted_token_ids = token_ids.copy()
        corrupted_token_ids[corruption_indices] = random_tokens
        assert len(corrupted_token_ids) == len(token_ids)
        return corrupted_token_ids

    
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, maxiters):
        self.warmup = warmup
        self.max_num_iters = maxiters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    
    
class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

class LossAwareSampler(ScheduleSampler):
    def __init__(self, num_timesteps, momentum=0.9, min_weight=1e-2):
        """
        Loss-aware importance sampling that dynamically adjusts weights based on loss history.

        :param num_timesteps: Total number of timesteps in the diffusion process.
        :param momentum: How much to smooth updates over time (higher = slower changes).
        :param min_weight: Minimum allowed weight to prevent zero probabilities.
        """
        self.num_timesteps = num_timesteps
        self.momentum = momentum
        self.min_weight = min_weight
        self.loss_ema = np.ones(num_timesteps, dtype=np.float32)  # Exponential moving average of loss

    def weights(self):
        """
        Compute the importance-sampling weights based on loss history.

        :return: A numpy array of weights (higher values = more likely to be sampled).
        """
        # Normalize losses so that probabilities are nonzero but biased by loss magnitude.
        w = np.maximum(self.loss_ema, self.min_weight)
        return w / np.sum(w)  # Normalize so they sum to 1

    def update_with_all_losses(self, ts, losses):
        """
        Update the loss-tracking weights based on the observed losses.

        :param ts: A list of timestep indices.
        :param losses: A list of corresponding loss values.
        """
        for t, loss in zip(ts, losses):
            self.loss_ema[t] = self.momentum * self.loss_ema[t] + (1 - self.momentum) * loss


class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        dataloader,
        batch_size,
        lr,
        max_iters, 
        weight_decay=0.0,
        gradient_clipping=1.,
        warmup = 1000, 
        finetune = False,
        tau = 400,
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        opt_params = None,
        use_scheduler = True
    ):
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.lr = lr
        #self.schedule_sampler = UniformSampler(diffusion)
        self.schedule_sampler = LossAwareSampler(diffusion.timesteps)
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.max_iters = max_iters
        self.finetune = finetune
        self.tau = tau

        self.step = 0
        self.resume_step = 0

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if opt_params is not None:
            self.opt.load_state_dict(opt_params)

        self.use_scheduler = use_scheduler
        self.scheduler = CosineWarmupScheduler(optimizer = self.opt, warmup=warmup, maxiters = max_iters)

        self.ddp_model = self.model
        self.device = device
        self.loss = 10


    def run_loop(self, num_epochs, path_to_checkpt=None):
        epoch = 0

        for epoch in range(num_epochs):
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                if self.step % 1000 == 0 and self.step!=0:
                    generated_smiles = []
                    with torch.no_grad():
                        for i in range(10):
                            generated_embeddings = [self.model.get_logits(x["sample"]).softmax(dim=-1).argmax(dim=-1).cpu() for x in self.diffusion.p_sample_loop_progressive(self.model, (30, 109, self.model.emb_dim))]
                            smiles = [[decode(process_decode(x.cpu().tolist(), [self.diffusion.vocab['<sos>'],self.diffusion.vocab['<eos>'],self.diffusion.vocab['<pad>']]), self.diffusion.reverse_vocab) for x in y] for y in generated_embeddings[-2:]]
                            generated_smiles.extend(smiles[-1])
                        
                        valids = round(sum(validity(generated_smiles))/len(generated_smiles)*100, 2)
                    logger.info(f"step {self.step}/{self.max_iters} noise_loss: {round(self.loss,6)}, validity: {valids}, lr: {self.scheduler.get_last_lr()[0]}")

                if self.step % 400 == 0 and self.step != 0:
                    logger.info(f"step {self.step}/{self.max_iters} noise_loss: {round(self.loss,6)}, lr: {self.scheduler.get_last_lr()[0]}")
                             
                if self.finetune:
                    self.run_step(batch[0], batch[-1])
                else:
                    self.run_step(batch[0])

                progress_bar.set_postfix(loss=self.loss, lr=float(self.scheduler.get_last_lr()[0]))
                self.step += 1
            epoch += 1

        checkpoint = { 
            'epoch': epoch,
            'model': self.ddp_model.state_dict(),
            'optimizer': self.opt.state_dict()}
        if path_to_checkpt is None:
            torch.save(checkpoint, 'checkpoint_new.pth')
        else:
            torch.save(checkpoint, path_to_checkpt) 

    def run_step(self, input_ids, corrupted_input_ids = None):
        self.forward_backward(input_ids, corrupted_input_ids)
        self.optimize_normal()


    def forward_backward(self, input_ids, corrupted_input_ids=None):

        self.opt.zero_grad()
        t, weights = self.schedule_sampler.sample(input_ids.shape[0], self.device)

        max_boost = 0.7
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            t,
            input_ids,
            corrupted_input_ids,
            self.tau
        )

        losses = compute_losses(boost_factor=max_boost*(self.step/self.max_iters))

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_all_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        loss.backward()

        self.loss = loss.item()

    def grad_clip(self):

        max_grad_norm=self.gradient_clipping #3.0
        # Revert to normal clipping otherwise, handling Apex or full precision

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_grad_norm,
        )

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
            
        self.opt.step()
        
        if self.use_scheduler:
            self.scheduler.step()


