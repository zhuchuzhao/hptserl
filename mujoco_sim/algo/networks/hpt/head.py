import jax  
import jax.numpy as jnp  
import flax.linen as nn  
from typing import List, Tuple, Dict, Any  
from .trunk import SimpleTransformer, MultiheadAttention  
  
class PolicyHead(nn.Module):  
    """基础策略头模块"""  
    input_dim: int  
    output_dim: int  
      
    def setup(self):  
        pass  
      
    def __call__(self, x, deterministic=True):  
        raise NotImplementedError("基类方法需要被子类实现")  
  
class MLPHead(PolicyHead):  
    """MLP策略头"""  
    hidden_dims: List[int] = (256, 256)  
    use_layernorm: bool = True  
    dropout: float = 0.0  
    tanh_output: bool = False  
      
    @nn.compact  
    def __call__(self, x, deterministic=True):  
        for i, hidden_dim in enumerate(self.hidden_dims):  
            x = nn.Dense(hidden_dim, name=f'dense_{i}')(x)  
            if self.use_layernorm:  
                x = nn.LayerNorm(name=f'ln_{i}')(x)  
            x = nn.silu(x)  
            if self.dropout > 0:  
                x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)  
          
        x = nn.Dense(self.output_dim, name='output')(x)  
        if self.tanh_output:  
            x = nn.tanh(x)  
          
        return x  
  
class TransformerDecoderHead(PolicyHead):  
    """基于Transformer Decoder的策略头"""  
    action_horizon: int = 1  
    crossattn_heads: int = 8  
    crossattn_dim_head: int = 64  
    dropout: float = 0.1  
      
    def setup(self):  
        # 初始化可学习的动作tokens  
        self.action_tokens = self.param('action_tokens',  
                                        nn.initializers.normal(0.02),  
                                        (1, self.action_horizon, self.output_dim))  
          
        # 跨模态注意力层  
        embed_dim = self.crossattn_dim_head * self.crossattn_heads  
          
        # 创建MLP网络  
        self.mlp = nn.Sequential([  
            nn.Dense(embed_dim),  
            nn.silu,  
            nn.Dense(self.output_dim)  
        ])  
          
        # 创建跨模态注意力  
        from mujoco_sim.algo.networks.hpt.scene_representation import CrossAttention  
        self.cross_attention = CrossAttention(  
            query_dim=self.output_dim,  
            heads=self.crossattn_heads,  
            dim_head=self.crossattn_dim_head,  
            dropout=self.dropout  
        )  
      
    def __call__(self, x, deterministic=True):  
        batch_size = x.shape[0]  
          
        # 处理上下文向量  
        context = self.mlp(x)  
        context = context.reshape(batch_size, -1, context.shape[-1])  
          
        # 复制动作tokens并应用跨模态注意力  
        queries = jnp.repeat(self.action_tokens, batch_size, axis=0)  
        out = self.cross_attention(  
            queries, context, deterministic=deterministic)  
          
        # 对于单步动作，降维  
        return out.reshape(batch_size, -1)