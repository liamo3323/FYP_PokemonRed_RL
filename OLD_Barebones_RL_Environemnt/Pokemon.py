import pokebase as pb

class pokemon:

    def __init__(self, name:str, level:int=5):
        self.name = pb.pokemon(name)
        self.id = self.name.id
        self.level = level
        self.base_experience = self.name.base_experience
        self.ability = self.name.abilities[0]
        self.held_item = None

        self.moves = [("",0),("",0),("",0),("",0)]
        self.learnable_moves_level = None
        self.learnable_moves_tm = None

        self.base_stats = [self.name.stats[0].base_stat,self.name.stats[1].base_stat,self.name.stats[2].base_stat,self.name.stats[3].base_stat,self.name.stats[4].base_stat,self.name.stats[5].base_stat]
        self.stats = []
        self.types = [self.name.types[0].type.name]

        self.setStats()
        self.getMoveList()

    def setStats(self):
        Hp = round(0.01*(2*self.base_stats[0]*self.level)+self.level+10)
        Atk = round(0.01*(2*self.base_stats[1]*self.level)+5)
        Def = round(0.01*(2*self.base_stats[2]*self.level)+5)
        SpAtk = round(0.01*(2*self.base_stats[3]*self.level)+5)
        SpDef = round(0.01*(2*self.base_stats[4]*self.level)+5)
        Speed = round(0.01*(2*self.base_stats[5]*self.level)+5)
        self.stats = [Hp,Atk,Def,SpAtk,SpDef,Speed]

    def getMoveList(self):
        moves = self.name.moves
        levelup_list=[]
        tm_list= []
        for x in moves:
            if x.version_group_details[0].move_learn_method.name == "level-up":
                levelup_list.append((x.version_group_details[0].level_learned_at, x.move.name))
            elif x.version_group_details[0].move_learn_method.name == "machine":
                tm_list.append((x.version_group_details[0].level_learned_at, x.move.name))

        levelup_list = sorted(levelup_list, key=lambda x:int(x[0]))
        tm_list = sorted(levelup_list, key=lambda x:int(x[0]))
        
        self.learnable_moves_level = levelup_list
        self.learnable_moves_tm = tm_list

    def sefLearnedMoves(self):

