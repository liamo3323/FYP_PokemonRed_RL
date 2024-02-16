from os.path import exists
from pathlib import Path
import uuid
from PkRed_env.red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


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

    algorithm = "DQN"
    batch_size = 512
    n_epochs = 1
    gamma = 0.999
    learn_steps = 64
    # <-- this is how long an episode is before it restarts to starting state!

    sess_path = Path(f'{algorithm}_session_{str(uuid.uuid4())[:8]}')
    num_cpu = 8  # 64 #46  # Also sets the number of episodes per training iteration
    ep_length = 2048 * 2

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False, 'reward_scale': 0.001,
        'action_freq': 24, 'init_state': 'RL_Training&Env/has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': 'RL_Training&Env/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'extra_buttons': False
    }

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
        model.learn(total_timesteps=(ep_length)*num_cpu *
                    1000, callback=checkpoint_callback)
