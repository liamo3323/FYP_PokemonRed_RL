from os.path import exists
from pathlib import Path
import datetime
import uuid
from PkRed_env.red_gym_env_v2 import RedGymEnv
from PkRed_env.stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO, A2C, DQN
from dopamine.agents.rainbow import rainbow_agent
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf), 
            stream_metadata = {
                "user": "RE", 
                "env_id": rank,
                "color": "#800080",
                "extra": "", # any extra text you put here will be displayed
            }
        )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def make_model(algorithm):
    if algorithm == ("PPO"):
        model = PPO('MultiInputPolicy', env, verbose=1, n_steps=ep_length,
                    batch_size=batch_size, n_epochs=n_epochs, gamma=gamma
                    , tensorboard_log=sess_path)

    elif algorithm == ("A2C"):
        model = A2C('MultiInputPolicy', env, verbose=1,
                    n_steps=ep_length, gamma=gamma, tensorboard_log=sess_path)

    elif algorithm == ("DQN"):
        model = DQN('MultiInputPolicy', env, verbose=1, gamma=gamma, tensorboard_log=sess_path)

    else:
        raise Exception('MISSING ALGORITHM!')

    return model

if __name__ == "__main__":
    current_datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S")
    use_wandb_logging = True
    algorithm = "PPO"
    batch_size = 128
    gamma = 0.998
    n_epochs = 3
    learn_steps = 32
    sess_id = str(uuid.uuid4())[:8]

    sess_path = Path(f'Sessions/{algorithm}_Session_{current_datetime_str}_{sess_id}_env2_1')
    # sess_path = Path(f'Sessions/PPO_Session_0307161249_7602f77b_env2_2')
    print(sess_path)

    num_cpu = 14  #! cannot go any higher than 12 <- also crashes after 3-4 hours
    episode_length_per_cpu = 1250 #? each episode will be 1250 steps long 
    ep_length = num_cpu * episode_length_per_cpu #? EPISODE LENGTH WILL BE 17,500
    total_timesteps = (ep_length)*1714 #? TOTAL TRAINING TIME WILL BE 30,000,000 (approx)

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': 'has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'reward_scale': 1.3, 'explore_weight': 2, 'battle_weight': 2,
                'use_screen_explore': True, 'extra_buttons': False, 'sim_frame_dist': 2_000_000.0,
            }
    
    print(env_config)

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix="poke")
    
    callbacks = [checkpoint_callback, TensorboardCallback()]

    # log to wandb
    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-red-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # put a checkpoint here you want to start from
    # file_name = f"Sessions/PPO_Session_0307161249_7602f77b_env2_1/poke_4235000_steps"
    file_name = f"blank"

    train_steps_batch = ep_length
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch #? Essentially save frequency
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print("\nloading new model!")
        model = make_model(algorithm)
        print(model.policy)

    print(f"training for {total_timesteps/num_cpu} steps...")

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), tb_log_name=f"poke_{algorithm}")

    if use_wandb_logging:
        current_time_str = datetime.datetime.now().strftime("%d%H%M")
        print(f"!!!!COMPLETED at {current_time_str}!!!")
        run.finish()