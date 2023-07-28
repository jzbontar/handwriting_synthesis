import math
import random
import time
import sys

import torch
import torch.nn as nn
import pickle
from torch.nn import functional as F

# hyperparameters
subsample = 4
batch_size = 64
max_epochs = 1000
eval_interval = 25
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
max_strokes_len = 224
wandb_log = False
wandb_project = 'handwriting_synthesis'
wandb_run_name = '_'.join(sys.argv[1:])
exec(open('configurator.py').read())

random.seed(1337)
torch.manual_seed(1337)

def plot_example(ax, text, strokes):
    ax.set_title(text)
    ax.axis('equal')
    xs, ys = [0], [0]
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

dataset = pickle.load(open(f'data/all_{subsample}.pkl', 'rb'))
dataset = list(dataset.values())

MU = torch.tensor([8.4637, 0.2108])
STD = torch.tensor([44.9969, 37.0469])

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

            # 0: pad
            # 1: start
            # 2: normal
            # 3: end stroke
            # 4: end example
            cls = s[:, 2].long() + 2
            cls[-1] = 4
            cls = torch.cat((torch.tensor([1]), cls))
            cls = cls.to(device)

            pos = s[:, :2]
            pos = (pos - MU) / STD
            pos = torch.cat((torch.zeros(1, 2), pos))
            pos = pos.to(device)

            flat_dataset.append(dict(line=l, cls=cls, pos=pos))
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
        line, cls_x, pos_x, cls_y, pos_y = [], [], [], [], []
        for ex in data[i:i + batch_size]:
            line.append(ex['line'])
            cls_x.append(ex['cls'][:-1])
            pos_x.append(ex['pos'][:-1])
            cls_y.append(ex['cls'][1:])
            pos_y.append(ex['pos'][1:])
        yield dict(
            line=nn.utils.rnn.pad_sequence(line, batch_first=True),
            cls_x=nn.utils.rnn.pad_sequence(cls_x, batch_first=True),
            pos_x=nn.utils.rnn.pad_sequence(pos_x, batch_first=True),
            cls_y=nn.utils.rnn.pad_sequence(cls_y, batch_first=True),
            pos_y=nn.utils.rnn.pad_sequence(pos_y, batch_first=True),
        )
        i += batch_size

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        loss, mse_loss, ce_loss = [], [], []
        for batch in get_batches(split):
            with ctx:
                d = model(batch)
            loss.append(d['loss'].item())
            mse_loss.append(d['mse_loss'].item())
            ce_loss.append(d['ce_loss'].item())
        out[split] = dict(
            loss=torch.tensor(loss).mean(),
            mse_loss=torch.tensor(mse_loss).mean(),
            ce_loss=torch.tensor(ce_loss).mean(),
        )
    model.train()
    return out

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, maxlen):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return token_embedding + self.pos_embedding[:token_embedding.size(1)]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = PositionalEncoding(n_embd, max_strokes_len)
        self.line_emb = nn.Embedding(128, n_embd)
        self.cls_emb = nn.Embedding(5, n_embd)
        self.cls_head = nn.Linear(n_embd, 5)
        self.pos_emb = nn.Linear(2, n_embd)
        self.pos_head = nn.Linear(n_embd, 2)
        self.transformer = nn.Transformer(
            d_model=n_embd,
            nhead=n_head,
            num_encoder_layers=n_enc_layer,
            num_decoder_layers=n_dec_layer,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, b):
        attn_mask = nn.Transformer.generate_square_subsequent_mask(b['cls_x'].size(1), device=device) == -torch.inf
        src_pad_mask = b['line'] == 0
        tgt_pad_mask = b['cls_x'] == 0

        line_emb = self.line_emb(b['line'])
        cls_emb = self.cls_emb(b['cls_x'])
        pos_emb = self.pos_emb(b['pos_x'])
        y = self.transformer(
            self.pos(line_emb), 
            self.pos(cls_emb + pos_emb), 
            tgt_mask=attn_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        pos = self.pos_head(y)
        cls = self.cls_head(y)
        if 'pos_y' not in b:
            return dict(pos=pos, cls=cls)
        mse_loss = F.mse_loss(pos[~tgt_pad_mask], b['pos_y'][~tgt_pad_mask])
        ce_loss = F.cross_entropy(cls[~tgt_pad_mask], b['cls_y'][~tgt_pad_mask])
        loss = mse_loss + ce_loss
        return dict(loss=loss, mse_loss=mse_loss, ce_loss=ce_loss)
    
@torch.no_grad()
def generate(text, max_tokens, temperature=1.0):
    line = encode(text)[None].to(device)
    pos_x = torch.zeros((1, 1, 2), device=device)
    cls_x = torch.ones((1, 1), dtype=torch.long, device=device)
    for _ in range(max_tokens):
        with ctx:
            d = model(dict(line=line, pos_x=pos_x, cls_x=cls_x))
        pos = torch.normal(d['pos'][0, -1], temperature)
        pos_x = torch.cat((pos_x, pos[None, None]), dim=1)
        probs = F.softmax(d['cls'][0, -1] / (temperature + 1e-5), dim=0)
        cls = torch.multinomial(probs, num_samples=1)
        cls_x = torch.cat((cls_x, cls[None]), dim=1)
        if cls.item() == 4:
            break
    pos_x = pos_x.cpu()
    cls_x = cls_x.cpu()
    pos_x = pos_x[0, 1:]
    cls_x = cls_x[0, 1:, None]
    pos_x = pos_x * STD + MU
    cls_x = cls_x == 3
    cls_x[-1] = True
    sample = torch.cat((pos_x, cls_x), dim=1)
    return sample
    
ctx = torch.autocast(device, getattr(torch, dtype))
model = Model()
model = model.to(device)
if compile:
    model = torch.compile(model)
torch.set_float32_matmul_precision('high')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if wandb_log:
    import wandb
    import pylab as plt
    wandb.init(project=wandb_project, name=wandb_run_name)

tt = time.time()
for epoch in range(max_epochs):
    for batch in get_batches('train'):
        with ctx:
            d = model(batch)
        optimizer.zero_grad(set_to_none=True)
        d['loss'].backward()
        optimizer.step()
    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        losses = estimate_loss()
        print(f"epoch {epoch}: train loss {losses['train']['loss']:.4f}, val loss {losses['val']['loss']:.4f}, time {time.time() - tt:.1f}")
        if wandb_log:
            data = {f'{split}/{loss}':losses[split][loss] for split in ('train', 'val') for loss in ('loss', 'mse_loss', 'ce_loss')}
            
            texts = ['Hello World', 'Katarina Zupancic', 'A MOVE to stop Mr . Gaitskell']
            fig, axs = plt.subplots(len(texts))
            for i, text in enumerate(texts):
                sample = generate(text, max_strokes_len, temperature=0.0)
                plot_example(axs[i], text, sample.cpu())
            fig.tight_layout()
            data['samples'] = wandb.Image(fig)
            plt.close()

            wandb.log(data=data, step=epoch)
            
        tt = time.time()
