from os.path import exists
from pathlib import Path
import uuid
from PkRed_env.red_gym_env import RedGymEnv
from stable_baselines3 import A2C, PPO
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
        # env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2048 * 10

    env_config = {
        'headless': True, 'save_final_state': True, 'early_stop': False, 'reward_scale': 0.001,
        'action_freq': 24, 'init_state': 'has_pokedex_nballs.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': True, 'fast_video': True, 'session_path': sess_path,
        'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True, 'extra_buttons': False
    }

    num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = make_env(0, env_config)()

    # env_checker.check_env(env)

    file_name = 'Sessions/PPO_Session_0223160349_env2/poke_23961600_steps'

    print('\nrunning pretrained model')
    print('\nloading checkpoint')
    model = PPO.load(file_name, env=env)

    # keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        action = 5  # pass action
        agent_enabled = True
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if truncated:
            break
    env.close()
    print('--Finished Demonstration--')
