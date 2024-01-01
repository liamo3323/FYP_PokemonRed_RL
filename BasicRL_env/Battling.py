from array import *

class battling:
    
    def damage_calc():
        Level   = 0
        Power   = 0
        Attack  = 0
        Defense = 0
        Critical= 0
        Random  = 0
        Stab    = 0
        Type    = 0
        Burn    = 0

        return ( ((((((2*Level)/5)+2)*Power*Attack/Defense)/50)+2) * Critical * Random * Stab * Type * Burn )

    def type_effectiveness(attacker: str, defender: str) -> float:

        # horizontal (columns Defense) | Vertical (rows Attack)
        typeChart = [[1,1,1,1,1,1,1,1,1,1,1,1,0.5,0,1,1,0.5,1],[1,0.5,0.5,1,2,2,1,1,1,1,1,2,0.5,1,0.5,1,2,1],[1,2,0.5,1,0.5,1,1,1,2,1,1,1,2,1,0.5,1,1,1],[1,1,2,0.5,0.5,1,1,1,0,2,1,1,1,1,0.5,1,1,1],[1,0.5,2,1,0.5,1,1,0.5,2,0.5,1,0.5,2,1,0.5,1,0.5,1],[1,0.5,0.5,1,2,0.5,1,1,2,2,1,1,1,1,2,1,0.5,1],[2,1,1,1,1,2,1,0.5,1,0.5,0.5,0.5,2,0,1,2,2,0.5],[1,1,1,1,2,1,1,0.5,0.5,1,1,1,0.5,0.5,1,1,0,2],[1,2,1,2,0.5,1,1,2,1,0,1,0.5,2,1,1,1,2,1],[1,1,1,0.5,2,1,2,1,1,1,1,2,0.5,1,1,1,0.5,1],[1,1,1,1,1,1,2,2,1,1,0.5,1,1,1,1,0,0.5,1],[1,0.5,1,1,2,1,0.5,0.5,1,0.5,2,1,1,0.5,1,2,0.5,0.5],[1,2,1,1,1,2,0.5,1,0.5,2,1,2,1,1,1,1,0.5,1],[0,1,1,1,1,1,1,1,1,1,2,1,1,2,1,0.5,1,1],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,0,0.5,0],[1,1,1,1,1,0.5,1,1,1,2,1,1,2,1,0.5,1,0.5,1],[1,0.5,0.5,0.5,1,2,1,1,1,1,1,1,2,1,1,1,0.5,2],[1,0.5,1,1,1,1,2,0.5,1,1,1,1,1,1,2,2,0.5,1]]

        def typeToNumber(typeName: str) -> int:
            if typeName == "Normal":
                typeNum = 0
            elif typeName == "Fire":
                typeNum = 1
            elif typeName == "Water":
                typeNum = 2
            elif typeName == "Grass":
                typeNum = 3   
            elif typeName == "Electric":
                typeNum = 4
            elif typeName == "Ice":
                typeNum = 5
            elif typeName == "Fighting":
                typeNum = 6
            elif typeName == "Poison":
                typeNum = 7  
            elif typeName == "Ground":
                typeNum = 8
            elif typeName == "Flying":
                typeNum = 9
            elif typeName == "Psychic":
                typeNum = 10
            elif typeName == "Bug":
                typeNum = 11   
            elif typeName == "Rock":
                typeNum = 12
            elif typeName == "Ghost":
                typeNum = 13
            elif typeName == "Dragon":
                typeNum = 14
            elif typeName == "Dark":
                typeNum = 15    
            elif typeName == "Steel":
                typeNum = 16  
            elif typeName == "Fairy":
                typeNum = 17
            else:
                # catch if there is a string error 
                typeNum = -1
            return typeNum

        atkTypeNum = typeToNumber(attacker)
        defTypeNum = typeToNumber(defender)

        # [Attacker][Defender]
        print(atkTypeNum)
        print(defTypeNum)
        return (typeChart[atkTypeNum][defTypeNum])

