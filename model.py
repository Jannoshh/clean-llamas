import math
from dataclasses import dataclass

import einops
import torch
from jaxtyping import Float, Int
from torch import nn, Tensor
from einops import einsum, rearrange


@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_q_heads: int = 12
    n_kv_heads: int = 4
    n_layers: int = 12


class RMSNorm(nn.Module):
    def __init__(self, d_model, norm_eps):
        super().__init__()
        self.norm_eps = norm_eps
        self.g = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(
            self,
            residual: Float[Tensor, "b posn d_model"]
    ) -> Float[Tensor, "b posn d_model"]:
        residual_std = (torch.var(residual, dim=-1, keepdim=True, unbiased=False) + self.norm_eps).sqrt()
        normalized = residual / residual_std
        out = normalized * self.g + self.b
        return out


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model, init_range):
        super().__init__()
        self.W_E = nn.Parameter(torch.empty((d_vocab, d_model)))
        nn.init.normal_(self.W_E, std=init_range)

    def forward(
            self,
            tokens: Int[Tensor, "b posn"]
    ) -> Float[Tensor, "b posn d_model"]:
        return self.W_E[tokens]


class Unembed(nn.Module):
    def __init__(self, d_model, d_vocab, init_range):
        super().__init__()
        self.W_U = nn.Parameter(torch.empty((d_model, d_vocab)))
        nn.init.normal_(self.W_U, std=init_range)
        self.b_U = nn.Parameter(torch.zeros(d_vocab, requires_grad=False))

    def forward(
            self,
            normalized_resid_final: Float[Tensor, "b position d_model"]
    ) -> Float[Tensor, "b position d_vocab"]:
        return einsum(normalized_resid_final, self.W_U, "b posn d_model, d_model d_vocab -> b posn d_vocab") + self.b_U


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_pos = nn.Parameter(torch.empty((cfg.max_posn, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.init_range)

    def forward(self, posns):
        # positions: [b, posn]
        embeddings = self.W_pos[posns]
        return embeddings


class GroupedMultiQueryAttention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, d_model, n_q_heads, n_kv_heads, d_head, init_range, device):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.W_Q = nn.Parameter(torch.empty((n_q_heads, d_model, d_head)))
        self.W_K = nn.Parameter(torch.empty((n_kv_heads, d_model, d_head)))
        self.W_V = nn.Parameter(torch.empty((n_kv_heads, d_model, d_head)))
        self.W_O = nn.Parameter(torch.empty((n_kv_heads, d_head, d_model)))
        nn.init.normal_(self.W_Q, std=init_range)
        nn.init.normal_(self.W_K, std=init_range)
        nn.init.normal_(self.W_V, std=init_range)
        nn.init.normal_(self.W_O, std=init_range)
        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device=device))

    def forward(
            self,
            normalized_resid_pre: Float[Tensor, "b posn d_model"]
    ) -> Float[Tensor, "b posn d_model"]:
        # Calculate query, key and value vectors
        q = einsum(normalized_resid_pre, self.W_Q,
                   "b posn d_model, n_q_heads d_model d_head ->  b posn n_q_heads d_head")
        k = einsum(normalized_resid_pre, self.W_K,
                   "b posn d_model, n_kv_heads d_model d_head -> b posn n_kv_heads d_head")
        v = einsum(normalized_resid_pre, self.W_V,
                   "b posn d_model, n_kv_heads d_model d_head -> b posn n_kv_heads d_head")

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        grouped_q = rearrange(
            q, "b posn (n_kv_heads g_size) d_head -> b posn g_size n_kv_heads d_head", n_kv_heads=self.n_kv_heads
        )
        attn_scores = einsum(grouped_q, k,
                             "b posn_q g_size n_kv_heads d_head, b posn_k n_kv_heads d_head -> b n_kv_heads posn_q posn_k")
        attn_scores_masked = self.apply_causal_mask(attn_scores / math.sqrt(self.d_head))
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einsum(v, attn_pattern,
                   "b posn_k n_kv_heads d_head, b n_kv_heads posn_q posn_k -> b posn_q n_kv_heads d_head")

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = einsum(z, self.W_O, "b posn_q n_kv_heads d_head, n_kv_heads d_head d_model -> b posn_q d_model")
        return attn_out

    def apply_causal_mask(
            self,
            attn_scores: Float[Tensor, "b n_heads query_pos key_pos"]
    ) -> Float[Tensor, "b n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        # Define a mask that is True for all positions we want to set probabilities to zero for
        all_ones = torch.ones(attn_scores.shape[2:], device=attn_scores.device)
        mask = torch.triu(all_ones, diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, d_model, d_mlp, init_range, device):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty((d_model, d_mlp)))
        self.W_out = nn.Parameter(torch.empty((d_mlp, d_model)))
        self.b_in = nn.Parameter(torch.zeros((d_mlp)))
        self.b_out = nn.Parameter(torch.zeros((d_model)))
        nn.init.normal_(self.W_in, std=init_range)
        nn.init.normal_(self.W_out, std=init_range)

    def forward(
            self,
            normalized_resid_mid: Float[Tensor, "b posn d_model"]
    ) -> Float[Tensor, "b posn d_model"]:
        pre = einsum(normalized_resid_mid, self.W_in, "b posn d_model, d_model d_mlp -> b posn d_mlp") + self.b_in
        post = nn.functional.relu(pre)  # TODO: SwiGLU
        mlp_out = einsum(post, self.W_out, "b posn d_mlp, d_mlp d_model -> b posn d_model") + self.b_out
        return mlp_out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_head, d_mlp, norm_eps, init_range, device):
        super().__init__()
        self.norm1 = RMSNorm(d_model, norm_eps)
        self.attention = GroupedMultiQueryAttention(d_model, n_q_heads, n_kv_heads, d_head, init_range, device)
        self.norm2 = RMSNorm(d_model, norm_eps)
        self.feed_forward = FeedForwardSwiGLU(d_model, d_mlp, init_range, device)

    def forward(
            self,
            resid_pre: Float[Tensor, "b posn d_model"]
    ) -> Float[Tensor, "b posn d_model"]:
        resid_mid = self.attention(self.norm1(resid_pre)) + resid_pre
        resid_post = self.feed_forward(self.norm2(resid_mid)) + resid_mid
        return resid_post


class LLamaTransformer(nn.Module):
    def __init__(self, d_vocab, d_model, n_q_heads, n_kv_heads, d_head, d_mlp, n_layers, norm_eps, init_range, device):
        super().__init__()
        self.embed = Embed(d_vocab, d_model, init_range)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_q_heads, n_kv_heads, d_head, d_mlp, norm_eps, init_range, device)
            for _ in range(n_layers)
        ])
        self.norm_final = RMSNorm(d_model, norm_eps)
        self.unembed = Unembed(d_model, d_vocab, init_range)

    def forward(
            self,
            tokens: Int[Tensor, "b posn"]
    ) -> Float[Tensor, "b posn d_model"]:
        # Calculate embeddings for tokens and positions
        token_embeddings = self.embed(tokens)

        residual = token_embeddings
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.norm_final(residual))
        return logits


if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LLamaTransformer(cfg.d_vocab, cfg.d_model, cfg.n_q_heads, cfg.n_kv_heads, cfg.d_head, cfg.d_mlp, cfg.n_layers, cfg.norm_eps,
                             cfg.init_range, device)
    model.to(device)
    print(model)
    # Test model
    batch = torch.randint(0, cfg.d_vocab, (1, cfg.n_ctx), device=device)
    logits = model(batch)
    print(logits.shape)
