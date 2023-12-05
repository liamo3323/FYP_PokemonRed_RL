from os.path import exists
from pathlib import Path
import uuid
from PkmnRedEnv import PkmnRedEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from argparse_pokemon import *

sess_path = f'session_{str(uuid.uuid4())[:8]}'

run_steps = 2048
runs_per_update = 6
updates_per_checkpoint = 4

args = get_args('pkmn_run.py', ep_length=run_steps, sess_path=sess_path)

env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False, 
                'action_freq': 24, 'init_state': 'init_episode.state', 'max_steps': run_steps,
                'print_rewards': True, 'save_video': True, 'session_path': sess_path,
                'gb_path': 'PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0
            }

env_config = change_env(env_config, args)
env = PkmnRedEnv(config=env_config)
env_checker.check_env(env)

# Env Setup complete

learn_steps = 4
file_name = 'poke_'
inference_only = False 
tot_timesteps=run_steps*runs_per_update*updates_per_checkpoint
tot_timesteps=10

model = PPO('CnnPolicy', env, verbose=1, n_steps=run_steps*runs_per_update, batch_size=128, n_epochs=3, gamma=0.98)

for i in range(learn_steps):
    model.learn(total_timesteps=tot_timesteps)
    model.save(sess_path / Path(file_name+str(i)))

