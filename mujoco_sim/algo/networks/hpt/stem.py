import jax  
import jax.numpy as jnp  
import flax.linen as nn  
from typing import List, Optional, Dict, Any  
  
class CrossAttention(nn.Module):  
    query_dim: int  
    heads: int = 8  
    dim_head: int = 64  
    dropout: float = 0.0  
  
    def setup(self):  
        inner_dim = self.dim_head * self.heads  
        self.scale = self.dim_head ** -0.5  
        self.to_q = nn.Dense(inner_dim, use_bias=False)  
        self.to_kv = nn.Dense(inner_dim * 2, use_bias=False)  
        self.to_out = nn.Dense(self.query_dim)  
        self.dropout_layer = nn.Dropout(rate=self.dropout)  
  
    def __call__(  
        self,  
        x: jnp.ndarray,  
        context: jnp.ndarray,  
        mask: Optional[jnp.ndarray] = None,  
        deterministic: bool = True  
    ) -> jnp.ndarray:  
        b, n, _ = x.shape  
        h = self.heads  
        q = self.to_q(x)  
        k, v = jnp.split(self.to_kv(context), 2, axis=-1)  
        q = q.reshape(b, n, h, -1).transpose(0, 2, 1, 3)  
        k = k.reshape(b, context.shape[1], h, -1).transpose(0, 2, 1, 3)  
        v = v.reshape(b, context.shape[1], h, -1).transpose(0, 2, 1, 3)  
        attn = jnp.matmul(q, k.transpose(0,1,3,2)) * self.scale  
        if mask is not None:  
            big_neg = -1e9  
            mask = mask[:, None, None, :]  
            attn = jnp.where(mask, attn, big_neg)  
        attn = nn.softmax(attn, axis=-1)  
        attn = self.dropout_layer(attn, deterministic=deterministic)  
        out = jnp.matmul(attn, v)  
        out = out.transpose(0,2,1,3).reshape(b, n, -1)  
        return self.to_out(out)  
  
class PolicyStem(nn.Module):  
    modality: str  
    embed_dim: int  
    crossattn_heads: int = 8  
    crossattn_dim_head: int = 64  
    crossattn_dropout: float = 0.1  
    token_num: int = 16  
  
    def setup(self):  
        self.tokens = self.param(  
            'tokens',  
            nn.initializers.normal(0.02),  
            (1, self.token_num, self.embed_dim)  
        )  
        self.cross_attention = CrossAttention(  
            query_dim=self.embed_dim,  
            heads=self.crossattn_heads,  
            dim_head=self.crossattn_dim_head,  
            dropout=self.crossattn_dropout  
        )  
        self.dropout = nn.Dropout(rate=self.crossattn_dropout)  
  
    def __call__(  
        self,  
        x: jnp.ndarray,  
        deterministic: bool = True  
    ) -> jnp.ndarray:  
        b = x.shape[0]  
        feat = self.process_modality(x)  
        feat = feat.reshape(b, -1, self.embed_dim)  
        tokens = jnp.repeat(self.tokens, b, axis=0)  
        out = self.cross_attention(tokens, feat, deterministic=deterministic)  
        return out  
  
    def process_modality(self, x: jnp.ndarray) -> jnp.ndarray:  
        raise NotImplementedError("process_modality must be implemented by subclass")  
  
class VisionStem(PolicyStem):  
    """视觉输入处理模块: 使用简单 CNN 提取特征并映射到 embed_dim"""  
  
    @nn.compact  
    def process_modality(self, x: jnp.ndarray) -> jnp.ndarray:  
        b, t, H, W, C = x.shape  
        x = x.reshape(b * t, H, W, C)  
        x = nn.Conv(features=64, kernel_size=(8,8), strides=(4,4), name='conv1')(x)  
        x = nn.relu(x)  
        x = nn.Conv(features=128, kernel_size=(4,4), strides=(2,2), name='conv2')(x)  
        x = nn.relu(x)  
        x = nn.Conv(features=self.embed_dim, kernel_size=(3,3), strides=(1,1), name='conv3')(x)  
        x = nn.relu(x)  
        x = x.reshape(b, t, -1)  
        return x  
  
class ProprioStem(PolicyStem):  
    state_dim: Optional[int] = None  
    hidden_dims: List[int] = (256, 256)  
  
    @nn.compact  
    def process_modality(self, x: jnp.ndarray) -> jnp.ndarray:  
        print(x.shape ,"111111111111111111111")
        b, t, _ = x.shape  
        x = x.reshape(b * t, -1)  
        for i, dim in enumerate(self.hidden_dims):  
            x = nn.Dense(dim, name=f'dense_{i}')(x)  
            x = nn.LayerNorm(name=f'ln_{i}')(x)  
            x = nn.silu(x)  
        x = nn.Dense(self.embed_dim, name='proj')(x)  
        x = x.reshape(b, t, self.embed_dim)  
        return x  
  
class LanguageStem(PolicyStem):  
    """语言输入处理模块: 传入嵌入，直接返回"""  
  
    @nn.compact  
    def process_modality(self, x: jnp.ndarray) -> jnp.ndarray:  
        return x
