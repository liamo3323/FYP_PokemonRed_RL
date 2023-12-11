from pyboy import PyBoy
import sys


pyboy = PyBoy('PokemonRed.gb', debugging=False,
                disable_input=False,
                window_type='SDL2',
                hide_window='--quiet' in sys.argv,)

pyboy.load_state(open("init_episode.state", "rb"))

while not pyboy.tick():
    pass
pyboy.stop()