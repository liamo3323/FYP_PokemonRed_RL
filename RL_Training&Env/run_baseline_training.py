from os.path import exists
from pathlib import Path
import uuid
import datetime
from stream_agent_env_wrapper import StreamWrapper
from PkRed_env.red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    current_datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S")

    algorithm = "PPO"
    batch_size = 128
    n_epochs = 3
    gamma = 0.997
    learn_steps = 32

    sess_path = Path(f'Sessions/{algorithm}_Session_{current_datetime_str}')
    print(sess_path)
    num_cpu = 16  
    ep_length = 2048 * 10

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': 'has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'extra_buttons': False, 'explore_weight': 3
    }

    print(env_config)

    env = RedGymEnv(env_config)

    env = StreamWrapper(
                env, 
                stream_metadata = { # All of this is part is optional
                    "user": "ReLiam", # choose your own username
                    "env_id": id, # environment identifier
                    "color": "#1033ff", # choose your color :)
                })

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                             name_prefix='poke')

    if algorithm == ("PPO"):
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length,
                    batch_size=batch_size, n_epochs=n_epochs, gamma=gamma)

    elif algorithm == ("A2C"):
        model = A2C('CnnPolicy', env, verbose=1,
                    n_steps=ep_length, gamma=gamma)

    elif algorithm == ("DQN"):
        model = DQN('CnnPolicy', env, verbose=1, gamma=gamma)

    else:
        # Don't! If you catch, likely to hide bugs.
        raise Exception('MISSING ALGORITHM!')

    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*5000, callback=checkpoint_callback)
