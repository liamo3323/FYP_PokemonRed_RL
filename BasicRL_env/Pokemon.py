import pokebase as pb

class pokemon:
    
    def __init__(self, name:str):
        self.name = pb.pokemon(name)
        self.id = self.name.id
        self.level = 1
        self.base_experience = self.name.base_experience
        self.ability = self.name.abilities[0]
        self.held_item = None
        self.moves = None
        self.base_stats = [self.name.stats[0].base_stat,self.name.stats[1].base_stat,self.name.stats[2].base_stat,self.name.stats[3].base_stat,self.name.stats[4].base_stat,self.name.stats[5].base_stat]
        self.stats = setStats()
        self.types = [self.name.types[0].type.name]

    def setStats(self):
        Hp = 0.01*(2*self.base_stats[0]*self.level)+self.level+10
        Atk = 0.01*(2*base_stats[1]*self.level)+5
        Def = 0.01*(2*base_stats[2]*self.level)+5
        SpAtk = 0.01*(2*base_stats[3]*self.level)+5
        SpDef = 0.01*(2*base_stats[4]*self.level)+5
        Speed = 0.01*(2*base_stats[5]*self.level)+5
        self.stats = [Hp,Atk,Def,SpAtk,SpDef,Speed]
