import flax.linen as nn  
import jax  
import jax.numpy as jnp  
from functools import partial  
from typing import Optional, Callable, Any  
  
from mujoco_sim.algo.vision.resnet_v1 import resnetv1_configs, ResNetEncoder  
  
class SelfAttention(nn.Module):  
    """Self-attention layer for transformer implementation."""  
    num_heads: int = 8  
    head_dim: int = 64  
    dropout_rate: float = 0.0  
    dtype: Any = jnp.float32  
  
    @nn.compact  
    def __call__(self, x, training=False):  
        batch_size, seq_len, hidden_dim = x.shape  
        inner_dim = self.num_heads * self.head_dim  
  
        # QKV projection  
        qkv = nn.Dense(inner_dim * 3, dtype=self.dtype, name="qkv")(x)  
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)  
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, heads, seq_len, head_dim)  
        q, k, v = qkv[0], qkv[1], qkv[2]  
  
        # Attention  
        scale = jnp.sqrt(self.head_dim).astype(self.dtype)  
        attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / scale  
        attn = nn.softmax(attn, axis=-1)  
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=not training)  
  
        # Output projection  
        out = jnp.matmul(attn, v)  
        out = jnp.transpose(out, (0, 2, 1, 3))  
        out = out.reshape(batch_size, seq_len, inner_dim)  
        out = nn.Dense(hidden_dim, dtype=self.dtype, name="out")(out)  
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not training)  
        return out  
  
class TransformerBlock(nn.Module):  
    """Transformer block with self-attention."""  
    dim: int  
    num_heads: int = 8  
    head_dim: int = 64  
    mlp_dim: int = 1024  
    dropout_rate: float = 0.0  
    dtype: Any = jnp.float32  
  
    @nn.compact  
    def __call__(self, x, training=False):  
        # Layer normalization and self-attention  
        y = nn.LayerNorm(dtype=self.dtype)(x)  
        y = SelfAttention(  
            num_heads=self.num_heads,  
            head_dim=self.head_dim,  
            dropout_rate=self.dropout_rate,  
            dtype=self.dtype  
        )(y, training=training)  
        x = x + y  
  
        # Layer normalization and MLP  
        y = nn.LayerNorm(dtype=self.dtype)(x)  
        y = nn.Dense(self.mlp_dim, dtype=self.dtype)(y)  
        y = nn.gelu(y)  
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)  
        y = nn.Dense(self.dim, dtype=self.dtype)(y)  
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not training)  
        return x + y  
  
class HPTResNetEncoder(nn.Module):  
    """  
    HPT ResNet Encoder with transformer for token processing  
    Combines ResNet stem with transformer for token processing  
    """  
    resnet_config: str = "resnetv1-10"  
    pooling_method: str = "spatial_learned_embeddings"  
    num_spatial_blocks: int = 8  
    bottleneck_dim: Optional[int] = 256  
    transformer_blocks: int = 3  
    num_heads: int = 8  
    head_dim: int = 32  
    mlp_dim: int = 512  
    dropout_rate: float = 0.1  
      
    @nn.compact  
    def __call__(self, observations, train=True, encode=True):  
        # Initialize ResNet stem from resnet_v1 configurations  
        resnet_stem = resnetv1_configs[self.resnet_config](  
            pooling_method=self.pooling_method,  
            num_spatial_blocks=self.num_spatial_blocks,  
            pre_pooling=True  
        )  
          
        # Get features from ResNet stem  
        x = resnet_stem(observations, train=train)  
          
        # The ResNet stem with pre_pooling=True returns features before pooling  
        # These are spatial features with shape [B, H, W, C]  
        # Reshape to tokens for transformer  
        batch_size = x.shape[0]  
        h, w, c = x.shape[1:]  
        tokens = x.reshape(batch_size, h * w, c)  
          
        # Apply transformer blocks  
        for i in range(self.transformer_blocks):  
            tokens = TransformerBlock(  
                dim=c,  
                num_heads=self.num_heads,  
                head_dim=self.head_dim,  
                mlp_dim=self.mlp_dim,  
                dropout_rate=self.dropout_rate  
            )(tokens, training=train)  
          
        # Global pooling and bottleneck projection  
        tokens = jnp.mean(tokens, axis=1)  # [B, C]  
          
        if self.bottleneck_dim is not None:  
            tokens = nn.Dense(self.bottleneck_dim)(tokens)  
            tokens = nn.LayerNorm()(tokens)  
            tokens = nn.tanh(tokens)  
              
        return tokens