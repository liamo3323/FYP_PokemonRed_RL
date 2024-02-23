from os.path import exists
from pathlib import Path
import datetime
from PkRed_env.red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO, A2C, DQN
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
                    batch_size=batch_size, n_epochs=n_epochs, gamma=gamma)

    elif algorithm == ("A2C"):
        model = A2C('MultiInputPolicy', env, verbose=1,
                    n_steps=ep_length, gamma=gamma)

    elif algorithm == ("DQN"):
        model = DQN('MultiInputPolicy', env, verbose=1, gamma=gamma)

    else:
        # Don't! If you catch, likely to hide bugs.
        raise Exception('MISSING ALGORITHM!')

    return model

if __name__ == "__main__":

    current_datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S")
    use_wandb_logging = False
    algorithm = "PPO"
    batch_size = 128
    n_epochs = 3
    gamma = 0.998
    learn_steps = 32
    sess_path = Path(f'Sessions/{algorithm}_Session_{current_datetime_str}_env2')
    print(sess_path)
    ep_length = 2048 * 10

    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': 'has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'reward_scale': 0.5, 'explore_weight': 1
            }
    
    print(env_config)
    
    num_cpu = 12  #! Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="poke")
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    # put a checkpoint here you want to start from
    file_name = "session_8d5a9983/poke_32768000_steps" #"session_9ff8e5f0/poke_21626880_steps"
    train_steps_batch = ep_length
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = make_model(algorithm)

    print(model.policy)

    model.learn(total_timesteps=(ep_length)*num_cpu*10000, callback=CallbackList(callbacks), tb_log_name="poke_ppo")

