from pyboy import PyBoy
pyboy = PyBoy('PokemonRed.gb')
while not pyboy.tick():
    pass
pyboy.stop()