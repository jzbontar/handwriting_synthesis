import pylab as plt
import torch

MU = torch.tensor([8.4637, 0.2108, 0])
STD = torch.tensor([44.9969, 37.0469, 1])

def plot_example(ax, strokes):
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