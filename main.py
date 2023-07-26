import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import pickle
import random

import torch
from torch import nn, optim
from torch.nn import functional as F
import pylab as plt
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('action')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--block-size', type=int, default=32)
parser.add_argument('--max-iters', type=int, default=5_000)
parser.add_argument('--eval-interval', type=int, default=500)
parser.add_argument('--learning-rate', type=float, default=3e-3)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--eval-iters', type=int, default=200)
parser.add_argument('--d-model', type=int, default=32)
parser.add_argument('--nhead', type=int, default=4)
parser.add_argument('--num-layers', type=int, default=4)
parser.add_argument('--temperature', type=float, default=0.1)
args = parser.parse_args()

torch.manual_seed(0)
random.seed(0)

def preprocess():
    dataset = {}
    for ascii_fname in tqdm(sorted(Path('data/ascii').glob('**/*.txt'))):
        if ascii_fname == Path('data/ascii/z01/z01-000/z01-000z.txt'):
            continue

        # get lines
        sequences = open(ascii_fname).read()
        sequences = sequences.replace(r'%%%%%%%%%%%', '\n')
        sequences = [i.strip() for i in sequences.split('\n')]
        lines = sequences[sequences.index('CSR:') + 2:]
        lines = [line.strip() for line in lines if line.strip()]
        
        # get strokes
        linestrokes_dir = Path(str(ascii_fname.parent).replace('ascii', 'lineStrokes'))
        if not linestrokes_dir.is_dir():
            continue
        linestroke_fnames = [f for f in sorted(linestrokes_dir.iterdir()) if f.stem.startswith(ascii_fname.stem + '-')]
        if not linestroke_fnames:
            continue
        strokes = []
        for linestroke_fname in linestroke_fnames:
            strokes.append(parse_linestroke_file(linestroke_fname))

        dataset[ascii_fname.stem] = dict(lines=lines, strokes=strokes)
        assert len(lines) == len(strokes)
    pickle.dump(dataset, open('data/all.pkl', 'wb'))

def parse_linestroke_file(fname):
    root = ET.parse(fname).getroot()
    example = []
    prev = None
    for stroke in root.iter('Stroke'):
        for point in stroke:
            x = int(point.get('x'))
            y = int(point.get('y'))
            if prev is not None:
                dx = x - prev[0]
                dy = y - prev[1]
                example.append([dx, dy, 0])
            prev = x, y
        if example:
            example[-1][2] = 1
    return torch.tensor(example, dtype=torch.float)

def plot_example(ex):
    plt.close()
    plt.title(ex['line'])
    xs, ys = [], []
    prev = 0, 0
    plt.gca().set_aspect('equal')
    for dx, dy, end in ex['strokes']:
        x = dx + prev[0]
        y = dy + prev[1]
        xs.append(x)
        ys.append(-y)
        prev = x, y
        if end:
            plt.plot(xs, ys)
            xs, ys = [], []
    plt.show()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_project = nn.Linear(3, args.d_model)
        self.output_project = nn.Linear(args.d_model, 3)
        self.position_embedding_table = nn.Embedding(args.block_size, args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.nhead, dim_feedforward=args.d_model * 4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)

    def forward(self, x, y=None):
        pos_emb = self.position_embedding_table(torch.arange(x.shape[1], device=args.device))
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=args.device)
        x = self.input_project(x) 
        x = x + pos_emb
        x = self.transformer_encoder(x, mask=mask, is_causal=True)
        x = self.output_project(x)
        if y is None:
            return x
        mse_loss = F.mse_loss(x[:,:,:2], y[:,:,:2])
        bce_loss = F.binary_cross_entropy_with_logits(x[:,:,2], y[:,:,2])
        loss = mse_loss + bce_loss
        return loss

MU = torch.tensor([8.4637, 0.2108, 0])
STD = torch.tensor([44.9969, 37.0469, 1])
def train():
    dataset = pickle.load(open('data/all.pkl', 'rb'))
    val_names = set(l.strip() for l in open('data/testset_v.txt'))

    def flatten(dataset):
        flat_dataset = []
        for ex in dataset:
            for l, s in zip(ex['lines'], ex['strokes']):
                s = s.to(args.device)
                s = (s - MU) / STD
                flat_dataset.append(dict(line=l, strokes=s))
        return flat_dataset
    
    tr = flatten([v for k, v in dataset.items() if k not in val_names])
    va = flatten([v for k, v in dataset.items() if k in val_names])

    def get_batch(split):
        data = tr if split == 'train' else va
        x, y = [], []
        for _ in range(args.batch_size):
            while True:
                ex = random.choice(data)
                if ex['strokes'].shape[0] > args.block_size:
                    break
            j = random.randrange(ex['strokes'].shape[0] - args.block_size)
            x.append(ex['strokes'][j:j + args.block_size])
            y.append(ex['strokes'][j + 1:j + args.block_size + 1])
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(args.eval_iters)
            for i in range(args.eval_iters):
                xb, yb = get_batch(split)
                loss = model(xb, yb)
                losses[i] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    model = Model()
    model = model.to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    wandb.init(project='handwriting_synthesis', config=args)
    tt = time.time()
    for iter in range(args.max_iters):
        if iter % args.eval_interval == 0 or iter == args.max_iters - 1:
            losses = estimate_loss()
            if iter > 0:
                wandb.log(losses, step=iter)
            print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}, time {int(time.time() - tt)}')
            tt = time.time()
        xb, yb = get_batch('train')
        loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), 'model.pt')

@torch.no_grad()
def generate():
    model = Model()
    model.load_state_dict(torch.load('model.pt'))
    x = torch.zeros((1, 1, 3))
    for i in range(512):
        x_cond = x[:,-args.block_size:]
        pred = model(x_cond)
        dxdy = torch.normal(pred[0, -1, :2], args.temperature)
        storke_end = torch.rand(1) < F.sigmoid(pred[0, -1, 2])
        sample = torch.cat((dxdy, storke_end))
        x = torch.cat((x, sample[None, None]), dim=1)
    x = x * STD + MU
    x[:, -1, 2] = 1
    plot_example(dict(line='SAMPLE', strokes=x[0]))

if __name__ == '__main__':
    globals()[args.action]()