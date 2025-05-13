import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

import functools
from torch.optim import AdamW

import regex
from abc import ABC, abstractmethod

import logging

from utils import decode, process_decode
from validate import validity

import random
import wandb

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


def getrandomnumber(numbers,k,weights=None):
    if k==1:
        return random.choices(numbers,weights=weights,k=k)[0]
    else:
        return random.choices(numbers,weights=weights,k=k)

class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len, vocab, corrupt=False, corrupt_ratio=0.1):
        self.smiles_list = smiles_list
        self.encode = tokenizer
        self.max_len = max_len
        self.stoi = vocab
        self.corrupt = corrupt
        self.corrupt_ratio = corrupt_ratio
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.rg = regex.compile(pattern)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        pure_tokens = self.encode(smi, self.stoi)

        tokens_with_symbols = [self.stoi['<sos>']] + pure_tokens + [self.stoi['<eos>']]
        pad_len = self.max_len - len(tokens_with_symbols)
        tokens_with_symbols = tokens_with_symbols + [self.stoi['<pad>']] * pad_len

        if self.corrupt:
            corrupted_tokens_with_symbols = self.corrupt_one(smi).squeeze(0)
        else:
            corrupted_tokens_with_symbols = torch.tensor([0] * len(tokens_with_symbols), dtype=torch.long)

        assert len(tokens_with_symbols) == len(corrupted_tokens_with_symbols), \
            f'{len(tokens_with_symbols)}, {len(corrupted_tokens_with_symbols)}'

        labels = self.label_tokens(tokens_with_symbols)
        return torch.tensor(tokens_with_symbols, dtype=torch.long), torch.tensor(labels, dtype=torch.long), corrupted_tokens_with_symbols

    def corrupt_one(self, smi):
        # res = [self.toktoid[i] for i in self.rg.findall(smi)]
        res = [i for i in self.rg.findall(smi)]
        total_length = len(res) + 2
        if total_length>self.max_len:
            encoded = self.encode(smi)
            return [self.stoi['<sos>']] + encoded + [self.stoi['<eos>']]
        ######################## start corruption ###########################
        r = random.random()
        if r<0.3:
            pa,ring = True,True
        elif r<0.65:
            pa,ring = True,False
        else:
            pa,ring = False,True
        #########################
        max_ring_num  = 1
        ringpos = []
        papos = []
        for pos,at in enumerate(res):
            if at=='(' or at==')':
                papos.append(pos)
            elif at.isnumeric():
                max_ring_num = max(max_ring_num,int(at))
                ringpos.append(pos)
        # ( & ) remove
        r = random.random()
        if r<0.3:
            remove,padd = True,True
        elif r<0.65:
            remove,padd = True,False
        else:
            remove,padd = False,True
        if pa and len(papos)>0:
            if remove:
                # remove pa
                n_remove = getrandomnumber([1,2,3,4],1,weights = [0.6,0.2,0.1,0.1])
                p_remove = set(random.choices(papos,weights=None,k=n_remove))
                total_length -= len(p_remove)
                for p in p_remove:
                    res[p]=None
                    # print('debug pa delete {}'.format(p))
        # Ring remove
        r = random.random()
        if r<0.3:
            remove,radd = True,True
        elif r<0.65:
            remove,radd = True,False
        else:
            remove,radd = False,True
        if ring and len(ringpos)>0:
            if remove:
                # remove ring
                n_remove = getrandomnumber([1,2,3,4],1,weights = [0.7,0.2,0.05,0.05])
                p_remove = set(random.choices(ringpos,weights=None,k=n_remove))
                total_length -= len(p_remove)
                for p in p_remove:
                    res[p]=None
                    # print('debug ring delete {}'.format(p))
        # ring add & ( ) add
        if pa:
            if padd:
                n_add = getrandomnumber([1,2,3],1,weights = [0.8,0.2,0.1])
                n_add = min(self.max_len-total_length,n_add)
                for _ in range(n_add):
                    sele = random.randrange(len(res)+1)
                    res.insert(sele, '(' if random.random()<0.5 else ')')
                    # print('debug pa add {}'.format(sele))
                    total_length += 1
        if ring:
            if radd:
                n_add = getrandomnumber([1,2,3],1,weights = [0.8,0.2,0.1])
                n_add = min(self.max_len-total_length,n_add)
                for _ in range(n_add):
                    sele = random.randrange(len(res)+1)
                    res.insert(sele, str(random.randrange(1,max_ring_num+1)))
                    # print('debug ring add {}'.format(sele))
                    total_length += 1

        ########################## end corruption ###############################
        # print('test:',res)
        # print('test:',''.join([i for i in res if i is not None]))

        res = [self.stoi[i] for i in res if i is not None]
        res = [self.stoi['<sos>']] + res + [self.stoi['<eos>']]
        if len(res) < self.max_len:
            res += [self.stoi['<pad>']]*(self.max_len-len(res))
        else:
            res = res[:self.max_len]
            res[-1] = [self.stoi['<eos>']]
        return torch.LongTensor([res])

    def label_tokens(self, token_ids):
        label_classes = {
            '(': 0,
            ')': 1
        }
        labels = [label_classes.get(token, 2) for token in token_ids]
        return labels
    
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
                if self.step % 1500 == 0 and self.step!=0:
                    generated_smiles = []
                    for i in range(10):
                        generated_embeddings = []
                        self.model.eval()

                        for x in self.diffusion.p_sample_loop_progressive(self.model, (30, 109, self.model.emb_dim), time_steps=1000):
                            with torch.no_grad():
                                generated_embeddings.append(self.model.get_logits(x["sample"]).softmax(dim=-1).argmax(dim=-1).cpu())
                        smiles = [[decode(process_decode(x.cpu().tolist(), [self.diffusion.vocab['<sos>'],self.diffusion.vocab['<eos>'],self.diffusion.vocab['<pad>']]), self.diffusion.reverse_vocab) for x in y] for y in generated_embeddings[-2:]]
                        generated_smiles.extend(smiles[-1])
                        
                        valids = round(sum(validity(generated_smiles))/len(generated_smiles)*100, 2)
                    logger.info(f"step {self.step}/{self.max_iters} noise_loss: {round(self.loss,6)}, validity: {valids}, lr: {self.scheduler.get_last_lr()[0]}")
                    wandb.log({'validity':valids})
                    self.model.train()

                if self.step % 20 == 0:
                    logger.info(f"step {self.step}/{self.max_iters} noise_loss: {round(self.loss,6)}, lr: {self.scheduler.get_last_lr()[0]}")
                    wandb.log({'loss':round(self.loss,6), 'lr': self.scheduler.get_last_lr()[0]})

                self.run_step(  input_ids = batch[0], 
                                token_labels = batch[1],
                                corrupted_input_ids = batch[-1])

                progress_bar.set_postfix(loss=self.loss, lr=float(self.scheduler.get_last_lr()[0]))
                self.step += 1

                if self.step%10000 == 0 and self.step!=0:
                    checkpoint = { 
                        'epoch': epoch,
                        'model': self.ddp_model.state_dict(),
                        'optimizer': self.opt.state_dict()}
                    torch.save(checkpoint, f'checkpoints/checkpoint_{self.step}.pth')
            epoch += 1

        checkpoint = { 
            'epoch': epoch,
            'model': self.ddp_model.state_dict(),
            'optimizer': self.opt.state_dict()}
        torch.save(checkpoint, f'checkpoints/checkpoint_last.pth')

    def run_step(self, input_ids, token_labels = None, corrupted_input_ids = None):
        self.forward_backward(input_ids, token_labels, corrupted_input_ids)
        self.optimize_normal()


    def forward_backward(self, input_ids, token_labels = None, corrupted_input_ids=None):

        self.opt.zero_grad()
        t, weights = self.schedule_sampler.sample(input_ids.shape[0], self.device)

        max_boost = 0.7
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            t,
            input_ids,
            token_labels,
            corrupted_input_ids,
        )

        losses = compute_losses()

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


