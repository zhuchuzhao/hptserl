from dataclasses import dataclass  
from typing import List, Dict, Any, Optional, Tuple  
  
@dataclass  
class CrossAttentionLatentConfig:  
    """跨模态注意力潜在表示配置"""  
    image: int = 16  
    state: int = 8  
    language: int = 8  
    proprio: int = 8  
  
@dataclass  
class ModalityStemConfig:  
    """模态处理模块配置"""  
    input_dim: int  
    output_dim: int  
    filter_size: int = 3  
    conv_dimension: int = 2  # 2D或3D卷积  
  
@dataclass  
class HPTStemConfig:  
    """HPT输入处理模块配置"""  
    modalities: List[str]
    modality_embed_dim: int = 256  
    crossattn_heads: int = 8  
    crossattn_dim_head: int = 32  
    crossattn_modality_dropout: float = 0.1  
    crossattn_latent: CrossAttentionLatentConfig = None  
    state_dim: int = None  
      
    # 各模态的特定配置  
    image: ModalityStemConfig = None  
    state: ModalityStemConfig = None  
    language: ModalityStemConfig = None  
    proprio: ModalityStemConfig = None  
  
@dataclass  
class HPTHeadConfig:  
    """HPT输出头配置"""  
    head_type: str = "mlp"  # "mlp" 或 "transformer_decoder"  
    action_dim: int = None  
    hidden_dims: List[int] = None  
    use_layernorm: bool = True  
    dropout: float = 0.0  
    tanh_output: bool = True  
    # 仅用于transformer_decoder  
    crossattn_heads: int = 8  
    crossattn_dim_head: int = 32  
  
def create_default_hpt_config(  
    image_keys: List[str] = ["image"],  
    state_dim: int = None,  
    action_dim: int = None,  
    embed_dim: int = 512,  
    num_blocks: int = 8,  
    num_heads: int = 8,  
    use_language: bool = False,  
    use_proprio: bool = True,  
):  
    """  
    创建默认的HPT配置  
      
    Args:  
        image_keys: 图像键列表  
        state_dim: 状态维度  
        action_dim: 动作维度  
        embed_dim: 嵌入维度  
        num_blocks: Transformer块数量  
        num_heads: 注意力头数量  
        use_language: 是否使用语言模态  
        use_proprio: 是否使用本体感觉模态  
      
    Returns:  
        HPT配置字典  
    """  
    # 确定模态列表  
    modalities = []  
    modalities.extend(image_keys)  
      
    if use_proprio and state_dim is not None:  
        modalities.append("proprio")  
      
    if use_language:  
        modalities.append("language")  
      
    # 创建跨模态注意力潜在表示配置  
    crossattn_latent = CrossAttentionLatentConfig()  
      
    # 创建各模态的特定配置  
    modality_configs = {}  
      
    # 图像模态配置  
    for image_key in image_keys:  
        modality_configs[image_key] = ModalityStemConfig(  
            input_dim=3,  # RGB通道  
            output_dim=embed_dim // 4,  
            filter_size=3,  
            conv_dimension=2,  
        )  
      
    # 本体感觉模态配置  
    if use_proprio and state_dim is not None:  
        modality_configs["proprio"] = ModalityStemConfig(  
            input_dim=state_dim,  
            output_dim=embed_dim // 4,  
            filter_size=1,  
            conv_dimension=1,  
        )  
      
    # 语言模态配置  
    if use_language:  
        modality_configs["language"] = ModalityStemConfig(  
            input_dim=768,  # 假设使用BERT/T5等模型的嵌入  
            output_dim=embed_dim // 4,  
            filter_size=1,  
            conv_dimension=1,  
        )  
      
    # 创建HPT输入处理模块配置  
    stem_spec = HPTStemConfig(  
        modalities=modalities,  
        modality_embed_dim=embed_dim // 4,  
        crossattn_heads=8,  
        crossattn_dim_head=embed_dim // 32,  
        crossattn_modality_dropout=0.1,  
        crossattn_latent=crossattn_latent,  
        state_dim=state_dim,  
        **modality_configs  
    )  
      
    # 创建HPT输出头配置  
    head_spec = HPTHeadConfig(  
        head_type="mlp",  
        action_dim=action_dim,  
        hidden_dims=[256, 256],  
        use_layernorm=True,  
        dropout=0.0,  
        tanh_output=True,  
    )  
      
    # 创建完整的HPT配置  
    hpt_config = {  
        "embed_dim": embed_dim,  
        "num_blocks": num_blocks,  
        "num_heads": num_heads,  
        "use_modality_embedding": True,  
        "token_postprocessing": "action_token",  
        "observation_horizon": 1,  
        "action_horizon": 1,  
        "stem_spec": stem_spec,  
        "head_spec": head_spec,  
        "image_keys": image_keys,  
    }  
      
    return hpt_config