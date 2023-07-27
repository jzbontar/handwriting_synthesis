import random
import time

import torch
import torch.nn as nn
import pickle
from torch.nn import functional as F

# hyperparameters
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda'
dtype = 'bfloat16'
compile = False
eval_iters = 30
n_embd = 32
n_head = 4
n_enc_layer = 4
n_dec_layer = 4
dropout = 0.0
max_line_len = 128
max_strokes_len = 768
exec(open('configurator.py').read())

random.seed(1337)
torch.manual_seed(1337)

dataset = pickle.load(open('data/all.pkl', 'rb'))
dataset = list(dataset.values())

MU = torch.tensor([8.4637, 0.2108, 0])
STD = torch.tensor([44.9969, 37.0469, 1])

def encode(s):
    return torch.tensor(list(s.encode('ascii')), dtype=torch.long)

def flatten(dataset):
    flat_dataset = []
    for ex in dataset:
        for l, s in zip(ex['lines'], ex['strokes']):
            if len(l) > max_line_len or s.size(0) > max_strokes_len:
                continue
            l = encode(l)
            l = l.to(device)
            s = (s - MU) / STD
            s = s.to(device)
            flat_dataset.append(dict(line=l, strokes=s))
    return flat_dataset

n = int(len(dataset) * 0.8)
train_data = flatten(dataset[:n])
val_data = flatten(dataset[n:])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    enc_x, dec_x, dec_y = [], [], []

    for ex in random.sample(data, batch_size):
        enc_x.append(ex['line'])
        dec_x.append(ex['strokes'][:-1])
        dec_y.append(ex['strokes'][1:])
    enc_x = nn.utils.rnn.pad_sequence(enc_x, batch_first=True)
    dec_x = nn.utils.rnn.pad_sequence(dec_x, batch_first=True)
    dec_y = nn.utils.rnn.pad_sequence(dec_y, batch_first=True)
    return enc_x, dec_x, dec_y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch = get_batch(split)
            with ctx:
                _, loss = model(*batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_pos = nn.Embedding(max_line_len, n_embd)
        self.dec_pos = nn.Embedding(max_strokes_len, n_embd)
        self.enc_emb = nn.Embedding(128, n_embd)
        self.dec_emb = nn.Linear(3, n_embd)
        self.dec_head = nn.Linear(n_embd, 3)
        self.transformer = nn.Transformer(
            d_model=n_embd,
            nhead=n_head,
            num_encoder_layers=n_enc_layer,
            num_decoder_layers=n_dec_layer,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, enc_x, dec_x, dec_y=None):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_x.size(1), device=device)
        src_key_padding_mask = torch.where(enc_x == 0, -torch.inf, 0.)
        tgt_key_padding_mask = torch.where(torch.all(dec_x == 0, dim=2), -torch.inf, 0.)

        enc_pos = self.enc_pos(torch.arange(enc_x.size(1), device=device))
        dec_pos = self.dec_pos(torch.arange(dec_x.size(1), device=device))
        enc_emb = self.enc_emb(enc_x)
        dec_emb = self.dec_emb(dec_x)

        y = self.transformer(
            enc_emb + enc_pos, 
            dec_emb + dec_pos, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        y = self.dec_head(y)

        if dec_y is None:
            return y
        mse_loss = F.mse_loss(y[:, :, :2], dec_y[:, :, :2])
        bce_loss = F.binary_cross_entropy_with_logits(y[:, :, 2], dec_y[:, :, 2])
        loss = mse_loss + bce_loss
        return y, loss
    
ctx = torch.autocast(device, getattr(torch, dtype))
model = Model()
model = model.to(device)
if compile:
    model = torch.compile(model)
torch.set_float32_matmul_precision('high')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tt = time.time()
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {time.time() - tt}")
        tt = time.time()
    batch = get_batch('train')
    with ctx:
        _, loss = model(*batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()