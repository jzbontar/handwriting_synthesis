import argparse
from pathlib import Path
import pylab as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle
import time
import random
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--block-size', type=int, default=8)
parser.add_argument('--batch-size', type=int, default=4)
args = parser.parse_args()

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

def prepare_dataset():
    def flatten(dataset):
        flat_dataset = []
        for ex in dataset:
            for l, s in zip(ex['lines'], ex['strokes']):
                flat_dataset.append(dict(line=l, strokes=s))
        return flat_dataset

    dataset = pickle.load(open('data/all.pkl', 'rb'))
    val_names = set(l.strip() for l in open('data/testset_v.txt'))
    tr = flatten([v for k, v in dataset.items() if k not in val_names])
    va = flatten([v for k, v in dataset.items() if k in val_names])
    return tr, va

def train():
    def get_batch(split):
        x, y = [], []
        data = tr if split == 'train' else va
        for i in range(args.batch_size):
            ex = data[i]
            j = random.randrange(ex['strokes'].shape[0] - args.block_size)
            x.append(ex['strokes'][j:j + args.block_size])
            y.append(ex['strokes'][j + 1:j + args.block_size + 1])
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y
    
    tr, va = prepare_dataset()
    get_batch('train')

train()