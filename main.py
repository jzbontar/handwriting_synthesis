import random
import time

import torch
import torch.nn as nn
import pickle
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda'
dtype = torch.float32
compile = False
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.0

random.seed(1337)
torch.manual_seed(1337)

dataset = pickle.load(open('data/all.pkl', 'rb'))
dataset = list(dataset.values())

MU = torch.tensor([8.4637, 0.2108, 0])
STD = torch.tensor([44.9969, 37.0469, 1])
def flatten(dataset):
    flat_dataset = []
    for ex in dataset:
        for l, s in zip(ex['lines'], ex['strokes']):
            s = (s - MU) / STD
            s = s.to(device)
            flat_dataset.append(dict(line=l, strokes=s))
    return flat_dataset

n = int(len(dataset) * 0.8)
train_data = flatten(dataset[:n])
val_data = flatten(dataset[n:])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    x, y = [], []
    for _ in range(batch_size):
        while True:
            ex = random.choice(data)
            if ex['strokes'].shape[0] > block_size:
                break
        j = random.randrange(ex['strokes'].shape[0] - block_size)
        x.append(ex['strokes'][j:j + block_size])
        y.append(ex['strokes'][j + 1:j + block_size + 1])
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_project = nn.Linear(3, n_embd)
        self.output_project = nn.Linear(n_embd, 3)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

    def forward(self, x, y=None):
        device = x.device
        pos_emb = self.position_embedding_table(torch.arange(x.shape[1], device=device))
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=device)
        x = self.input_project(x) 
        x = x + pos_emb
        x = self.transformer_encoder(x, mask=mask, is_causal=True)
        x = self.output_project(x)
        if y is None:
            return x
        mse_loss = F.mse_loss(x[:,:,:2], y[:,:,:2])
        bce_loss = F.binary_cross_entropy_with_logits(x[:,:,2], y[:,:,2])
        loss = mse_loss + bce_loss
        return x, loss

ctx = torch.autocast(device, dtype)
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
    xb, yb = get_batch('train')
    with ctx:
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()