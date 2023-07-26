from dataclasses import dataclass
from torch import nn
import torch
import torch.nn.functional as F

@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_project = nn.Linear(3, config.n_embd)
        self.output_project = nn.Linear(config.n_embd, 3)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.n_embd, nhead=config.n_head, dim_feedforward=4 * config.n_embd, dropout=config.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)

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

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return 0
