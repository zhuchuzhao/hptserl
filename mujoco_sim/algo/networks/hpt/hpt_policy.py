import jax  
import jax.numpy as jnp  
import flax.linen as nn  
from typing import List, Dict, Any  
  
from .trunk import SimpleTransformer  
from .stem import VisionStem, ProprioStem, LanguageStem  
from .head import MLPHead  
from .config import HPTStemConfig, HPTHeadConfig  
  
class HPTPolicy(nn.Module):  
    """HPT策略模型的JAX实现"""  
    # 核心配置  
    stem_spec: HPTStemConfig  
    head_spec: HPTHeadConfig  
    image_keys: List[str]  
  
    # 可选超参  
    embed_dim: int = 1024  
    num_blocks: int = 24  
    num_heads: int = 16  
    observation_horizon: int = 4  
    action_horizon: int = 1  
    use_modality_embedding: bool = True  
    token_postprocessing: str = "action_token"  
  
    def setup(self):  
        # 创建 Transformer 主干  
        self.trunk = SimpleTransformer(  
            embed_dim=self.embed_dim,  
            num_blocks=self.num_blocks,  
            num_heads=self.num_heads  
        )  
  
        # 本地收集字典，避免对已冻结属性做就地修改  
        m_tokens: Dict[str, jnp.ndarray] = {}  
        a_tokens: Dict[str, jnp.ndarray] = {}  
        stems: Dict[str, nn.Module] = {}  
        heads: Dict[str, nn.Module] = {}  
  
        # 初始化各模态处理模块和模态 token  
        for modality in self.stem_spec.modalities:  
            key = f"{modality}"  
            print("22222222",modality)
            if "image" in modality:  
                module = VisionStem(  
                    modality=modality,  
                    embed_dim=self.stem_spec.modality_embed_dim,  
                    crossattn_heads=self.stem_spec.crossattn_heads,  
                    crossattn_dim_head=self.stem_spec.crossattn_dim_head,  
                    crossattn_dropout=self.stem_spec.crossattn_modality_dropout,  
                    token_num=getattr(self.stem_spec.crossattn_latent, modality)  
                )  
            elif "state" in modality or "proprio" in modality:  
                module = ProprioStem(  
                    modality=modality,  
                    embed_dim=self.stem_spec.modality_embed_dim,  
                    state_dim=self.stem_spec.state_dim,  
                    crossattn_heads=self.stem_spec.crossattn_heads,  
                    crossattn_dim_head=self.stem_spec.crossattn_dim_head,  
                    crossattn_dropout=self.stem_spec.crossattn_modality_dropout,  
                    token_num=getattr(self.stem_spec.crossattn_latent, modality)  
                )  
            elif "language" in modality:  
                module = LanguageStem(  
                    modality=modality,  
                    embed_dim=self.stem_spec.modality_embed_dim,  
                    crossattn_heads=self.stem_spec.crossattn_heads,  
                    crossattn_dim_head=self.stem_spec.crossattn_dim_head,  
                    crossattn_dropout=self.stem_spec.crossattn_modality_dropout,  
                    token_num=getattr(self.stem_spec.crossattn_latent, modality)  
                )  
            else:  
                raise ValueError(f"Unsupported modality: {modality}")  
  
            stems[key] = module  
            m_tokens[modality] = self.param(  
                f'modality_token_{modality}',  
                nn.initializers.normal(0.02),  
                (1, 1, self.stem_spec.modality_embed_dim)  
            )  
  
        # 初始化动作 token  
        if self.token_postprocessing == "action_token":  
            a_tokens[""] = self.param(  
                f'action_token',  
                nn.initializers.normal(0.02),  
                (1, self.action_horizon, self.embed_dim)  
            )  
  
        # 初始化策略头  
        if self.head_spec.head_type == "mlp":  
            heads[""] = MLPHead(  
                input_dim=self.embed_dim,  
                output_dim=self.head_spec.action_dim,  
                hidden_dims=self.head_spec.hidden_dims,  
                use_layernorm=self.head_spec.use_layernorm,  
                dropout=self.head_spec.dropout,  
                tanh_output=self.head_spec.tanh_output  
            )  
        elif self.head_spec.head_type == "transformer_decoder":  
            from .policy_head import TransformerDecoderHead  
            heads[""] = TransformerDecoderHead(  
                input_dim=self.embed_dim,  
                output_dim=self.head_spec.action_dim,  
                action_horizon=self.action_horizon,  
                crossattn_heads=self.head_spec.crossattn_heads,  
                crossattn_dim_head=self.head_spec.crossattn_dim_head,  
                dropout=self.head_spec.dropout  
            )  
        else:  
            raise ValueError(f"Unsupported head type: {self.head_spec.head_type}")  
  
        # 一次性赋值，避免修改 FrozenDict  
        self.modalities_tokens = m_tokens  
        self.action_tokens = a_tokens  
        self.stems = stems  
        self.heads = heads  
  
    def get_position_embedding(self, seq_len: int) -> jnp.ndarray:  
        def _get_sinusoid_encoding_table(n_position, d_hid):  
            positions = jnp.arange(n_position)  
            channels = jnp.arange(d_hid)  
            freq = 1 / (10000 ** (channels // 2 * 2.0 / d_hid))  
            pos_angles = positions[:, None] * freq[None, :]  
            pos_encoding = jnp.zeros((n_position, d_hid))  
            pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(pos_angles[:, 0::2]))  
            pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(pos_angles[:, 1::2]))  
            return pos_encoding[None, :, :]  
        return _get_sinusoid_encoding_table(seq_len, self.embed_dim)  
  
    def preprocess_tokens(
        self,
        domain: str,
        features: List[jnp.ndarray],
        deterministic: bool = True
    ) -> jnp.ndarray:
        # Guard against empty features
        if len(features) == 0:
            raise ValueError(
                f"HPTPolicy.preprocess_tokens: no features to concatenate. "
                f"Expected modalities {self.stem_spec.modalities}, but received no valid data for domain '{domain}'."
            )
        tokens = jnp.concatenate(features, axis=1)
        if self.token_postprocessing == "action_token":
            # Use the first action token if multiple keys exist
            action_array = next(iter(self.action_tokens.values()))
            batch_size = tokens.shape[0]
            action_tokens = jnp.repeat(action_array, batch_size, axis=0)
            tokens = jnp.concatenate([tokens, action_tokens], axis=1)
        position_tokens = self.get_position_embedding(tokens.shape[1])
        return tokens + position_tokens  
  
    def postprocess_tokens(self, trunk_tokens: jnp.ndarray) -> jnp.ndarray:  
        if self.token_postprocessing == "mean":  
            return jnp.mean(trunk_tokens, axis=1)  
        if self.token_postprocessing == "action_token":  
            return trunk_tokens[:, -self.action_horizon:]  
        if self.token_postprocessing == "max":  
            return jnp.max(trunk_tokens, axis=1)  
        if self.token_postprocessing == "last":  
            return trunk_tokens[:, -1]  
        raise ValueError(f"Unsupported token_postprocessing: {self.token_postprocessing}")  
  
    def stem_process(  
        self,  
        domain: str,  
        data: Dict[str, jnp.ndarray],  
        deterministic: bool = True  
    ) -> List[jnp.ndarray]:  
        features: List[jnp.ndarray] = []  
        for modality in self.stem_spec.modalities:  
            if modality not in data:  
                continue  
            stem = self.stems[f"{modality}"]  
            features.append(stem(data[modality], deterministic=deterministic))  
        return features  
  
    def __call__(  
        self,  
        domain: str,  
        data: Dict[str, jnp.ndarray],  
        deterministic: bool = True  
    ) -> jnp.ndarray:  
        stem_tokens = self.stem_process(domain, data, deterministic)  
        trunk_tokens = self.preprocess_tokens(domain, stem_tokens, deterministic)  
        trunk_tokens = self.trunk(trunk_tokens, deterministic=deterministic)  
        features = self.postprocess_tokens(trunk_tokens)  
        return self.heads["" ](features, deterministic=deterministic)
