# Make a world that is 50x50, on a 250x250 display.
import mesa

from src.model import Beach
from .portrayal import beach_portrayal
from .agents import NUMBER_OF_CELLS

SIZE_OF_CANVAS_IN_PIXELS_X = 500
SIZE_OF_CANVAS_IN_PIXELS_Y = 500

simulation_params = {
    "height": NUMBER_OF_CELLS, 
    "width": NUMBER_OF_CELLS,
    "n_LWRobots": mesa.visualization.Slider(
        'number of robots',
        5, #default
        2, #min
        10, #max
        1, #step
        "choose how many robots to include in the simulation"
    ),
    
    "n_cleaning_tanks": mesa.visualization.Slider(
        'number of robots',
        2, #default
        1, #min
        3, #max
        1, #step
        "choose how many robots to include in the simulation"
    ),
#! new line added below    
    "added_functionality": mesa.visualization.Checkbox(name="Added Functionality", value=False, description="Add further functionailty to simulation"),
    
    "rubbish_collection_robots": mesa.visualization.Checkbox(name="Rubbish Collection Robots", value=False, description="Add collection robots so cleaning robots are not required to empty themselves")

    }
grid = mesa.visualization.CanvasGrid(beach_portrayal, NUMBER_OF_CELLS, NUMBER_OF_CELLS, SIZE_OF_CANVAS_IN_PIXELS_X, SIZE_OF_CANVAS_IN_PIXELS_Y)


server = mesa.visualization.ModularServer(
    Beach, [grid], "Beach Clean", simulation_params
)