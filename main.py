import random
import time
import sys

import torch
import torch.nn as nn
import pickle
from torch.nn import functional as F

# hyperparameters
batch_size = 64
max_epochs = 100
eval_interval = 5
learning_rate = 3e-4
device = 'cuda'
dtype = 'bfloat16'
compile = False
n_embd = 32
n_head = 4
n_enc_layer = 4
n_dec_layer = 4
dropout = 0.0
max_line_len = 128
max_strokes_len = 768
wandb_log = False
wandb_project = 'handwriting_synthesis'
wandb_run_name = '_'.join(sys.argv[1:])
exec(open('configurator.py').read())

random.seed(1337)
torch.manual_seed(1337)

def plot_example(ax, text, strokes):
    ax.set_title(text)
    ax.axis('equal')
    xs, ys = [], []
    prev = 0, 0
    for dx, dy, end in strokes:
        x = dx + prev[0]
        y = dy + prev[1]
        xs.append(x)
        ys.append(-y)
        prev = x, y
        if end:
            ax.plot(xs, ys)
            xs, ys = [], []

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
            s = torch.cat((torch.zeros(1, 3), s))
            s = s.to(device)
            flat_dataset.append(dict(line=l, strokes=s))
    return flat_dataset
    
n = int(len(dataset) * 0.8)
train_data = flatten(dataset[:n])
val_data = flatten(dataset[n:])

def get_batches(split):
    if split == 'train':
        random.shuffle(train_data)

    data = train_data if split == 'train' else val_data
    i = 0
    while i + batch_size < len(data):
        enc_x, dec_x, dec_y = [], [], []
        for ex in data[i:i + batch_size]:
            enc_x.append(ex['line'])
            dec_x.append(ex['strokes'][:-1])
            dec_y.append(ex['strokes'][1:])
        enc_x = nn.utils.rnn.pad_sequence(enc_x, batch_first=True)
        dec_x = nn.utils.rnn.pad_sequence(dec_x, batch_first=True, padding_value=-1)
        dec_y = nn.utils.rnn.pad_sequence(dec_y, batch_first=True, padding_value=-1)
        yield enc_x, dec_x, dec_y            
        i += batch_size

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for batch in get_batches(split):
            with ctx:
                _, loss = model(*batch)
            losses.append(loss.item())
        out[split] = torch.tensor(losses).mean()
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
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_x.size(1), device=device) == -torch.inf
        src_key_padding_mask = enc_x == 0
        tgt_key_padding_mask = torch.all(dec_x == -1, dim=2)

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
        y_valid = y[~tgt_key_padding_mask]
        dec_y_valid = dec_y[~tgt_key_padding_mask]
        mse_loss = F.mse_loss(y_valid[:, :2], dec_y_valid[:, :2])
        bce_loss = F.binary_cross_entropy_with_logits(y_valid[:, 2], dec_y_valid[:, 2])
        loss = mse_loss + bce_loss
        return y, loss
    
ctx = torch.autocast(device, getattr(torch, dtype))
model = Model()
model = model.to(device)
if compile:
    model = torch.compile(model)
torch.set_float32_matmul_precision('high')

@torch.no_grad()
def generate(text, max_tokens, temperature=1.0):
    x = torch.zeros((1, 1, 3), device=device)
    enc_x = encode(text)[None].to(device)
    for _ in range(max_tokens):
        with ctx:
            pred = model(enc_x, x)[0, -1]
        dxdy = torch.normal(pred[:2], temperature)
        stroke_end = torch.rand(1, device=device) < F.sigmoid(pred[2])
        sample = torch.cat((dxdy, stroke_end))
        x = torch.cat((x, sample[None, None]), dim=1)
    return x[0, 1:]

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if wandb_log:
    import wandb
    import pylab as plt
    wandb.init(project=wandb_project, name=wandb_run_name)
tt = time.time()
for epoch in range(max_epochs):
    for batch in get_batches('train'):
        with ctx:
            _, loss = model(*batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        losses = estimate_loss()
        print(f"epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time {time.time() - tt}")
        if wandb_log:
            texts = ['Hello World', 'Katarina Zupancic', 'Jure Zbontar']
            fig, axs = plt.subplots(len(texts))
            for i, text in enumerate(texts):
                sample = generate(text, 512, temperature=0.2)
                plot_example(axs[i], text, sample.cpu())
            fig.tight_layout()
            wandb.log(data=dict(
                train=losses['train'],
                val=losses['val'],
                samples=wandb.Image(fig),
            ), step=epoch)
        tt = time.time()
