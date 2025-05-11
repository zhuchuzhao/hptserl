# mujoco_sim/algo/networks/hpt/transformer.py  
import jax  
import jax.numpy as jnp  
import flax.linen as nn  
from typing import Callable, List, Optional  
  
class MultiheadAttention(nn.Module):  
    """JAX实现的多头注意力机制"""  
    embed_dim: int  
    num_heads: int = 8  
    dropout: float = 0.0  
      
    def setup(self):  
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"  
        self.head_dim = self.embed_dim // self.num_heads  
        self.scale = self.head_dim ** -0.5  
          
        self.qkv = nn.Dense(features=self.embed_dim * 3, use_bias=True)  
        self.proj = nn.Dense(features=self.embed_dim)  
        self.dropout_layer = nn.Dropout(rate=self.dropout)  
      
    def __call__(self, x, attn_mask=None, deterministic=True):  
        batch_size, seq_len, _ = x.shape  
          
        # 生成查询、键和值  
        qkv = self.qkv(x)  
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)  
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]  
          
        q, k, v = qkv[0], qkv[1], qkv[2]  
          
        # 注意力计算  
        attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * self.scale  
          
        if attn_mask is not None:  
            big_neg = -1e9  
            attn = jnp.where(attn_mask, attn, big_neg)  
          
        attn = jax.nn.softmax(attn, axis=-1)  
        attn = self.dropout_layer(attn, deterministic=deterministic)  
          
        # 输出投影  
        out = jnp.matmul(attn, v)  
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)  
        out = self.proj(out)  
          
        return out  
  
class MLP(nn.Module):  
    """前馈网络模块"""  
    hidden_dim: int  
    output_dim: int  
    dropout: float = 0.0  
      
    @nn.compact  
    def __call__(self, x, deterministic=True):  
        x = nn.Dense(features=self.hidden_dim)(x)  
        x = nn.gelu(x)  
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)  
        x = nn.Dense(features=self.output_dim)(x)  
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)  
        return x  
  
class TransformerBlock(nn.Module):  
    """Transformer块"""  
    dim: int  
    num_heads: int  
    mlp_ratio: int = 4  
    dropout: float = 0.0  
    drop_path: float = 0.0  
      
    @nn.compact  
    def __call__(self, x, attn_mask=None, deterministic=True):  
        # 自注意力层  
        residual = x  
        x = nn.LayerNorm(epsilon=1e-6)(x)  
        x = MultiheadAttention(  
            embed_dim=self.dim,   
            num_heads=self.num_heads,   
            dropout=self.dropout  
        )(x, attn_mask, deterministic=deterministic)  
          
        # DropPath (随机深度) - 类似于PyTorch中的实现  
        if self.drop_path > 0.0 and not deterministic:  
            keep_prob = 1.0 - self.drop_path  
            mask = jax.random.bernoulli(  
                self.make_rng('dropout'), keep_prob, (x.shape[0], 1, 1))  
            mask = jnp.broadcast_to(mask, x.shape)  
            x = x * mask / keep_prob  
          
        x = residual + x  
          
        # MLP层  
        residual = x  
        x = nn.LayerNorm(epsilon=1e-6)(x)  
        x = MLP(  
            hidden_dim=int(self.dim * self.mlp_ratio),  
            output_dim=self.dim,  
            dropout=self.dropout  
        )(x, deterministic=deterministic)  
          
        # DropPath  
        if self.drop_path > 0.0 and not deterministic:  
            keep_prob = 1.0 - self.drop_path  
            mask = jax.random.bernoulli(  
                self.make_rng('dropout'), keep_prob, (x.shape[0], 1, 1))  
            mask = jnp.broadcast_to(mask, x.shape)  
            x = x * mask / keep_prob  
          
        x = residual + x  
        return x  
  
class SimpleTransformer(nn.Module):  
    """简单的Transformer编码器"""  
    embed_dim: int  
    num_blocks: int  
    num_heads: int  
    mlp_ratio: int = 4  
    dropout: float = 0.0  
    drop_path: float = 0.0  
      
    @nn.compact  
    def __call__(self, tokens, attn_mask=None, deterministic=True):  
        # 预处理层(可选)  
        x = tokens  
          
        # Transformer块序列  
        for i in range(self.num_blocks):  
            drop_path_rate = self.drop_path * i / self.num_blocks  
            x = TransformerBlock(  
                dim=self.embed_dim,  
                num_heads=self.num_heads,  
                mlp_ratio=self.mlp_ratio,  
                dropout=self.dropout,  
                drop_path=drop_path_rate  
            )(x, attn_mask, deterministic=deterministic)  
          
        # 后处理层(可选)  
        return x