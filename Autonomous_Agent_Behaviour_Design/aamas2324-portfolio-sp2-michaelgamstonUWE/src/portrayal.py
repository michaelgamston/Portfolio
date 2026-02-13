from src.agents import LWRobot, PalmTree, LWPlastic, Bin, HeavyPlastic, CleaningTank, ChargingStation, CompactedPlastic, RubishRobot, FULL, BATTERY_EMPTY


def beach_portrayal(agent):
    """
    Determine which portrayal to use according to the type of agent.
    """
    
    if isinstance(agent, LWRobot): 
        return lwrobot_portrayal(agent)
    elif isinstance(agent, PalmTree):
        return palm_tree_portrayal(agent)
    elif isinstance(agent, LWPlastic):
        return lwplastic_portrayal(agent)
    elif isinstance(agent, Bin):
        return bin_portrayal(agent)
    elif isinstance(agent, HeavyPlastic):
        return heavy_plastic_portrayal(agent)
    elif isinstance(agent, CleaningTank):
        return cleaning_tank_portrayal(agent)
    elif isinstance(agent, ChargingStation):
        return charging_station_portrayal(agent)
    elif isinstance(agent, CompactedPlastic):
        return compact_plastic_portrayal(agent)
    elif isinstance(agent, RubishRobot):
        return rubbish_collection_robot_portrayal(agent)    


def lwrobot_portrayal(robot):
    def get_hopper_colour(hopper_fullness):
        if hopper_fullness > 74:
            return "Red"
        # elif hopper_fullness > 69:
        #     return "#f0be0a"
        else: 
            return "Blue"

    if robot is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": robot.x,
        "y": robot.y,
        "Color": get_hopper_colour(robot.hopper_fullness),
    }
    
def palm_tree_portrayal(tree):
    if tree is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": tree.x,
        "y": tree.y,
        "Color": "Green",
    }
    
def lwplastic_portrayal(plastic):
    if plastic is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 0.1,
        "h": 0.1,
        "Filled": "true",
        "Layer": 0,
        "x": plastic.x,
        "y": plastic.y,
        "Color": "#fc5a03",
    }
    
def bin_portrayal(bin):
    if bin is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": bin.x,
        "y": bin.y,
        "Color": "Gray",
    }
    

def charging_station_portrayal(charger):
    if charger is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "x": charger.x,
        "y": charger.y,
        "Color": "#03fc77",
    }
    

def heavy_plastic_portrayal(plastic):
    if plastic is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 0.7,
        "h": 0.7,
        "Filled": "true",
        "Layer": 0,
        "x": plastic.x,
        "y": plastic.y,
        "Color": "#fc5a03",
    }


def cleaning_tank_portrayal(robot):
    def get_colour(robot):
        if robot.state == BATTERY_EMPTY:
            return "#03fc77"
        elif robot.state == FULL:
            return 'Red'
        else: 
            return "Black"
        
    if robot is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 1.2,
        "h": 1.2,
        "Filled": "true",
        "Layer": 0,
        "x": robot.x,
        "y": robot.y,
        "Color": get_colour(robot),
    }
    

def compact_plastic_portrayal(plastic):
    
    if plastic is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 0.5,
        "h": 0.5,
        "Filled": "true",
        "Layer": 0,
        "x": plastic.x,
        "y": plastic.y,
        "Color": "#fc5a03",
    }
    
    
def rubbish_collection_robot_portrayal(robot):
    
    if robot is None:
        raise AssertionError
    return {
        "Shape": "rect",
        "w": 0.7,
        "h": 0.7,
        "Filled": "true",
        "Layer": 0,
        "x": robot.x,
        "y": robot.y,
        "Color": "Grey",
    }