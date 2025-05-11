import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict
from mujoco_sim.algo.networks.hpt.hpt_policy import HPTPolicy

class HPTEncoder(nn.Module):
    """HPT编码器封装，用于与SACAgent集成"""
    hpt_config: Dict[str, Any]
    domain_name: str = "serl"

    def setup(self):
        # 创建 HPT 策略模型
        self.hpt_model = HPTPolicy(**self.hpt_config)

    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        train: bool = False,
        stop_gradient: bool = False
    ) -> jnp.ndarray:
        """
        编码观察，返回特征表示

        Args:
            observations: 观察字典
            train: 是否训练模式（影响 Dropout）
            stop_gradient: 是否对输出应用 stop_gradient

        Returns:
            编码后的特征
        """
        deterministic = not train
        processed_data = self._process_observations(observations)
        stem_tokens = self.hpt_model.stem_process(
            self.domain_name, processed_data, deterministic
        )
        trunk_tokens = self.hpt_model.preprocess_tokens(
            self.domain_name, stem_tokens, deterministic
        )
        trunk_tokens = self.hpt_model.trunk(
            trunk_tokens, deterministic=deterministic
        )
        features = self.hpt_model.postprocess_tokens(trunk_tokens)
        if stop_gradient:
            features = jax.lax.stop_gradient(features)
        return features

    def _process_observations(
        self,
        observations: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """处理 SACAgent 传入的观察，转换为 HPT 模型期望的格式"""
        processed_data: Dict[str, jnp.ndarray] = {}
        stem_modalities = self.hpt_config["stem_spec"].modalities

        # 图像模态
        for key in self.hpt_config.get("image_keys", []):
            if key in observations:
                processed_data[key] = observations[key]

        # 本体感觉/状态模态：HPT 使用 'proprio' 语义
        if "proprio" in stem_modalities and "state" in observations:
            processed_data["proprio"] = observations["state"]
        # 或者纯粹“state”模态
        elif "state" in stem_modalities and "state" in observations:
            processed_data["state"] = observations["state"]

        # 语言模态
        if "language" in stem_modalities and "language" in observations:
            processed_data["language"] = observations["language"]

        return processed_data
