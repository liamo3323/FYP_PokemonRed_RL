# Multi-Objective Reinforcement Learning w/ Pokemon Red

> Implementation code for dissertation 'Reinforcement Learning in Pokémon Red to Explore Multi-Objective Environments'

![](example_good.mp4)

## Getting Started

Windows:

1. Make sure Microsoft C++ Build Tools are installed.
2. Follow MacOS & Linux steps below...

**MacOS & Linux:**

1. Make sure Python is installed, version 3.11 is recommended.
2. Have a copy of the Pokemon Red ROM into the root of the directory. It should be called `PokemonRed.gb`
3. Install all python dependencies:

```sh
pip install -r requirements.txt
```

4. Run:

```sh
python run_baselines_PPO.py
```

This should start trianing a fresh model for 40 million steps

## Usage example

### Previewing a Pre-trained Model

1. Open `run_pretrianed_v2.py` and specify the model you want to load.
2. replace `file_name` with the path to the trained model (40 million step pretrained models are stored in `40mil_Saved_Sessions`)

```python
file_name = '40mil_Saved_Sessions/PPO-1_68694384/poke_40278000_steps'
```

3. Run:

```sh
python run_pretrained_v2.py
```

### Training a new model

1. Run anyone of the `run_baseline_(DQN/PPO/QRDQN)` files

```sh
python run_baseline_PPO
```

2. Training of the agent can be watched here [pwhiddy-visualizer](https://pwhiddy.github.io/pokerl-map-viz/)

## Wandb

All trained models metrics and graphs are saved onto Wandb.

Edit the values below to save them to your own wandb project

```python
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
```

Reward function tuning training runs can be viewed: [Reward-Tuning](https://wandb.ai/liam3323/pokemon-red-train)

Pre-trained 40 million steps can be viewed: [40-mil-steps](https://wandb.ai/liam3323/FYP-RL-DATA)

## Credits

Liam O'Driscoll

Distributed under the MIT license. See `LICENSE` for more information.

[https://github.com/FYP_PokemonRed_RL](https://github.com/liamo3323/FYP_PokemonRed_RL)
