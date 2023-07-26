from dataclasses import dataclass
import inspect
from torch import nn
import torch
import torch.nn.functional as F

import extern

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
        self.config = config
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.config.block_size:]
            pred = self(x_cond)
            dxdy = torch.normal(pred[0, -1, :2], temperature)
            storke_end = torch.rand(1, device=x.device) < F.sigmoid(pred[0, -1, 2])
            sample = torch.cat((dxdy, storke_end))
            x = torch.cat((x, sample[None, None]), dim=1)
        x = x.to('cpu')
        x = x * extern.STD + extern.MU
        x[:, -1, 2] = 1
        return x