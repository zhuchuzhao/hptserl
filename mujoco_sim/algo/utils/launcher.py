# !/usr/bin/env python3

import jax
from jax import nn
import jax.numpy as jnp

from agentlace.trainer import TrainerConfig

from mujoco_sim.algo.common.typing import Batch, PRNGKey
from mujoco_sim.algo.common.wandb import WandBLogger
from mujoco_sim.algo.agents.continuous.bc import BCAgent
from mujoco_sim.algo.agents.continuous.sac import SACAgent
from mujoco_sim.algo.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from mujoco_sim.algo.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from mujoco_sim.algo.vision.data_augmentations import batched_random_crop

##############################################################################


def make_bc_agent(
    seed, 
    sample_obs, 
    sample_action, 
    image_keys=("image",), 
    encoder_type="resnet-pretrained"
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_sac_pixel_agent_hybrid_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentHybridSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def make_sac_pixel_agent_hybrid_dual_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
):
    agent = SACAgentHybridDualArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000


    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value
    
def make_batch_augmentation_func(image_keys) -> callable:

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations
    
    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch
    
    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "algo",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger

def make_sac_hpt_agent(  
    seed,  
    sample_obs,  
    sample_action,  
    image_keys=("image",),  
    embed_dim=512,  
    num_blocks=8,  
    num_heads=8,  
    reward_bias=0.0,  
    target_entropy=None,  
    discount=0.97,  
):  
    """  
    创建一个使用HPT编码器的SAC代理的便捷函数  
      
    Args:  
        seed: 随机种子  
        sample_obs: 样本观察  
        sample_action: 样本动作  
        image_keys: 图像键列表  
        embed_dim: HPT嵌入维度  
        num_blocks: HPT Transformer块数量  
        num_heads: HPT注意力头数量  
        reward_bias: 奖励偏置  
        target_entropy: 目标熵  
        discount: 折扣因子  
      
    Returns:  
        SAC代理  
    """  
    from mujoco_sim.algo.networks.hpt.config import create_default_hpt_config  
    from mujoco_sim.algo.agents.continuous.sac import make_sac_agent_with_hpt  
      
    # 获取状态维度  
    state_dim = None  
    if "state" in sample_obs:  
        state_dim = sample_obs["state"].shape[-1]  
      
    # 获取动作维度  
    action_dim = sample_action.shape[-1]  
      
    # 创建HPT配置  
    hpt_config = create_default_hpt_config(  
        image_keys=image_keys,  
        state_dim=state_dim,  
        action_dim=action_dim,  
        embed_dim=embed_dim,  
        num_blocks=num_blocks,  
        num_heads=num_heads,  
        use_language=False,  
        use_proprio=True,  
    )  
      
    # 创建SAC代理  
    agent = make_sac_agent_with_hpt(  
        jax.random.PRNGKey(seed),  
        sample_obs,  
        sample_action,  
        hpt_config=hpt_config,  
        domain_name="serl",  
        policy_kwargs={  
            "tanh_squash_distribution": True,  
            "std_parameterization": "exp",  
            "std_min": 1e-5,  
            "std_max": 5,  
        },  
        critic_network_kwargs={  
            "activations": nn.tanh,  
            "use_layer_norm": True,  
            "hidden_dims": [256, 256],  
        },  
        policy_network_kwargs={  
            "activations": nn.tanh,  
            "use_layer_norm": True,  
            "hidden_dims": [256, 256],  
        },  
        temperature_init=1e-2,  
        discount=discount,  
        backup_entropy=False,  
        critic_ensemble_size=2,  
        critic_subsample_size=None,  
        reward_bias=reward_bias,  
        target_entropy=target_entropy,  
        augmentation_function=make_batch_augmentation_func(image_keys),  
    )  
      
    return agent