from os.path import exists
from pathlib import Path
import uuid
from PkRed_env.red_gym_env_v2 import RedGymEnv
from stable_baselines3 import PPO, A2C, DQN
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

    sess_path = Path(f'Sessions/PPO_session_{str(uuid.uuid4())[:8]}_Pretrained')
    ep_length = 2000

    env_config = {
                'headless': False, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': 'has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': True, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
            }

    # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = make_env(0, env_config)()

    # env_checker.check_env(env)

    file_name = '40mil_Saved_Sessions/PPO-1_68694384/poke_40278000_steps'
    print('\nloading checkpoint pretrained model')
    model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    # keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        agent_enabled = True
        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if truncated:
            break
    env.close()
    print('--Finished Demonstration--')
