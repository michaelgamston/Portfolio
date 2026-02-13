import mesa
import numpy as np
from src.agents import LWRobot, PalmTree, LWPlastic, Bin, HeavyPlastic, CleaningTank, ChargingStation, CompactedPlastic, RubishRobot
from src.agents import SEARCHING, FINISHED, FULL, MOVING, READY
import random
import math
            
def get_overall_plastic_level(model) -> int: 
    """return the number of all plastics 

    Args:
        model (Beach): Simulation Model
    Returns:
        int: numebr of plastic peices
    """
    overall_plastic = [a for a in model.schedule.agents if isinstance(a, LWPlastic) or isinstance(a, HeavyPlastic) or isinstance(a, CompactedPlastic)]
    return len([a for a in overall_plastic if a.picked == False])

def get_light_plastic_level(model) -> int:
    """returns the number of light plastics

    Args:
        model (Beach): Simulation Model

    Returns:
        int: number of light plastics
    """
    return len([a for a in model.schedule.agents if isinstance(a, LWPlastic) and a.picked == False])

def get_heavy_plastic_level(model) -> int:
    """returns the number of heavy plastic peices

    Args:
        model (Beach): Simulation Model

    Returns:
        int: number of heavly plastics 
    """
    return len([a for a in model.schedule.agents if isinstance(a, HeavyPlastic) and a.picked == False])

def get_compact_plastic_level(model) -> int:
    """returns the number of compacted plastic peices

    Args:
        model (Beach): Simulation Model

    Returns:
        int: number of compact plastics 
    """
    return len([a for a in model.schedule.agents if isinstance(a, CompactedPlastic) and a.picked == False])


def is_a_robot(agent) -> bool: 
    """return true if the agent is a robot 

    Args:
        agent (mesa.agent)

    Returns:
        bool
    """
    if isinstance(agent, LWRobot) or isinstance(agent, CleaningTank) or isinstance(agent, RubishRobot):
        return True
    else: 
        return False
    

def get_agent_id(agent) -> int:
    """return the id of a given agent

    Args:
        agent (robot): a robot in the simulation

    Returns:
        int: id
    """
    return agent.unique_id


def get_agent_type(agent) -> str: 
    """returns the type of a given agent

    Args:
        agent (robot): a robot in the simulation

    Returns:
        str: robot type
    """
    return agent.__class__.__name__

def get_agent_state(agent) -> str:
    """returns the state of a given agent 

    Args:
        agent (robot): a robot in the simulation

    Returns:
        str: state
    """
    if is_a_robot(agent):
        match agent.state:
            case 1: 
                return "SEARCHING"
            case 2: 
                return "FINISHED"
            case 3: 
                return "FULL"
            case 4: 
                return "MOVING"
            case 5: 
                return "AVOIDANCE"
            case 6: 
                return "BATTERY_EMPTY"
            case 7: 
                return "READY"  
    else: 
        return "N/A"
        
        
#!added the below class
class Bid():
    """A class to store bids
    """
    def __init__(self, tank : CleaningTank, bid : int) -> None:
        self.tank = tank
        self.bid = bid

class Beach(mesa.Model):
    """ Model representing an automated beach clean"""

    def __init__(self, n_LWRobots=10, n_palm_trees = 6, n_heavy_plastic = 15, n_cleaning_tanks = 3, width=50, height=50, added_functionality=False, rubbish_collection_robots=False):
        """
            * Set schedule defining model activation
            * Sets the number of robots as per user input
            * Sets the grid space of the model
            * Create n Robots as required and place them randomly on the edge of the left side of the 2D space.
            * Create m Boxes as required and place them randomly within the model (Hint: To simplify you can place them in the same horizontal position as the Robots). Make sure robots or boxes do not overlap with each other.
        """
        self.tick = 0
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.width = width
        self.height = height
    
        self.n_LWRobots = n_LWRobots
        self.grid = mesa.space.MultiGrid(width, height, torus=True)
        
#!added the below
        self.added_functionality = added_functionality
        self.heavy_plastic_maximum_spwan_rate = 600
        self.heavy_plastic_minimum_spwan_rate = 1000
        self.heavy_plastic_current_spwan_rate = self.heavy_plastic_minimum_spwan_rate
        
        self.rubbish_collection_robots = rubbish_collection_robots
        self.rubbish_collection_robots_object_list = []
        self.compacted_plastic_locations = []
        
#! added the below 
        self.cleaning_tank_jobs_list = set()
        self.cleaning_tank_object_list = []
        self.cleaning_tank_plastic_tracker = [()]*n_cleaning_tanks
        
        #place the bin randomly
        bin_id = n_LWRobots+n_palm_trees
        bin_loc = (np.random.choice(np.arange(width)), np.random.choice(np.arange(height))) 
        bin = Bin(bin_id, self, bin_loc)
        self.grid.place_agent(bin, bin_loc)
#! moved the bin spwan to below the bin the robots
        
        #add charging station randomly 
        id_sum = n_LWRobots+n_palm_trees +1 
        while True: 
            if self.grid.is_cell_empty(charger_loc := (np.random.choice(np.arange(width)), np.random.choice(np.arange(height)))):
                break
        charger = ChargingStation(id_sum, self, charger_loc)
        self.grid.place_agent(charger, charger_loc)
        self.charger_loc = charger_loc
        
        id_sum = 0
        #find how many rows each robot will search
        y_step = round(height / n_LWRobots)
        #find middle twenty squares ranomly spawn robots here 4*5 rectangle    
        lwr_x_spawn = [x for x in range((round(width/2)-3), (round(width/2)+2))]
        lwr_y_spawn= [y for y in range((round(height/2)-2), (round(height/2)+2))]
        for n in range(n_LWRobots):
            
            #choose spwan location
            while(True):
                pos = (random.choice(lwr_x_spawn), random.choice(lwr_y_spawn))
                if self.grid.is_cell_empty(pos):
                    break
            
            #find search coordinates 
            y_start = n * y_step    
            search_coordinate = []  
            for y in range(y_start, (y_step + y_start)): 
                x_range = np.arange(0, width)
                if y % 2:
                    search_coordinate.extend([(x, y) for x in np.flip(x_range)])
                else:
                    search_coordinate.extend([(x, y) for x in x_range])

            lwr = LWRobot(n+id_sum, self, pos, search_coordinate, bin_loc)    
            self.schedule.add(lwr)
            self.grid.place_agent(lwr, pos)   
            
        #find empty cells
        empty_cells = []
        for contents, cell in self.grid.coord_iter():
            if not contents:
                empty_cells.append(cell)
        
        #add cleaning tanks
        id_sum = n_LWRobots     
        for n in range(n_cleaning_tanks): 
            pos = empty_cells.pop(np.random.choice(np.arange(len(empty_cells))))
            cleaning_tank = CleaningTank(n+id_sum, self, pos, charger_loc, bin_loc, (width, height))
            self.schedule.add(cleaning_tank)
            self.grid.place_agent(cleaning_tank, pos)
#! added the below
            self.cleaning_tank_object_list.append(cleaning_tank)
            
        #spwan in rubbish robots          
        id_sum = n_LWRobots + n_cleaning_tanks
        if self.rubbish_collection_robots: 
            for n in range(3):
                rubbish_robot = RubishRobot(id_sum, self, bin_loc, bin_loc)
                self.schedule.add(rubbish_robot)
                self.grid.place_agent(rubbish_robot, bin_loc) 
                id_sum+=1 
                self.rubbish_collection_robots_object_list.append(rubbish_robot)
                
        #spwan in bin 
        
        
        
        #randomly add palm trees
        
        for n in range(n_palm_trees):
            
            
            pos = empty_cells.pop(np.random.choice(np.arange(len(empty_cells))))
            palm_tree = PalmTree(n+id_sum, self, pos)
            self.grid.place_agent(palm_tree, pos)
            
        
                
        #add heavy plastic 
        id_sum = n_LWRobots+n_palm_trees+2+ n_cleaning_tanks
        for n in range(n_heavy_plastic):
            chosen_cell = empty_cells.pop(np.random.choice(np.arange(len(empty_cells))))
            lw_plastic = HeavyPlastic((n+id_sum), self, chosen_cell)
            self.schedule.add(lw_plastic)
            self.grid.place_agent(lw_plastic, chosen_cell)
        
        #places plastic on 80% of the grid 
        id_sum = n_LWRobots+n_palm_trees+n_heavy_plastic+2+ n_cleaning_tanks
        for n in range(int((width*height)*0.8)):
            chosen_cell = empty_cells.pop(np.random.choice(np.arange(len(empty_cells))))
            lw_plastic = LWPlastic((n+id_sum), self, chosen_cell)
            self.schedule.add(lw_plastic)
            self.grid.place_agent(lw_plastic, chosen_cell)
    
        self.last_id = n + 1 + id_sum             

        self.running = True
        
        self.datacollector = mesa.DataCollector(
            
            model_reporters= {
                "overall_plastic": get_overall_plastic_level, 
                "light_plastic": get_light_plastic_level,
                "heavy_plastic": get_heavy_plastic_level,
                "compact_plastic" : get_compact_plastic_level
                },
            
            agent_reporters = {
                "id" : get_agent_id, 
                "type" : get_agent_type, 
                "state" : get_agent_state
            }
        )


#! added below function 
    def add_compacted_plastic(self, pos : tuple) -> None: 
        compact_plastic = CompactedPlastic(self.last_id, self, pos)
        self.last_id+=1
        self.schedule.add(compact_plastic)
        self.grid.place_agent(compact_plastic, (pos))
        self.compacted_plastic_locations.append(pos)
        
        
    def assign_rubbish_robots_jobs(self) -> None: 
            for robot in self.rubbish_collection_robots_object_list:
                if len(self.compacted_plastic_locations) > 0:
                    if robot.state == READY:
                        print(f"Assigned rubbish robot {str(robot.unique_id)} the job at {str(self.compacted_plastic_locations[0])}")
                        robot.assign_new_job(self.compacted_plastic_locations.pop(0))
        
    
    def auction_jobs(self) -> None:
        print("auctionging the following jobs - ",end="")
        print(self.cleaning_tank_jobs_list)
        if self.cleaning_tank_jobs_list:
            #find all tanks that can be in the auction 
            bidders = [tank for tank in self.cleaning_tank_object_list if tank.state == MOVING]
            jobs_taken = []
            for job in self.cleaning_tank_jobs_list: 
                
                bids = []
                if len(bidders) > 1: 
                    print("more than one bidder")
                    for tank in bidders:
                        bids.append(Bid(tank, tank.bid(job)))
                elif len(bidders) == 1 : 
                    print("one bidder")
                    if job not in self.cleaning_tank_plastic_tracker:
                        bidders[0].assign_new_job(job)
                    else:
                        print("job already taken")
                    jobs_taken.append(job)
                    break
                else: 
                    print("no bidders")
                    break
                
                #find the best pip will only reach here if there are multiple bidders
                max_bid = bids[0].bid
                winning_bid = bids[0]
                for bid in bids[1:]: 
                    if bid.bid > max_bid:
                        max_bid = bid.bid
                        winning_bid = bid
                        
                if job not in self.cleaning_tank_plastic_tracker:        
                    winning_bid.tank.assign_new_job(job)
                    jobs_taken.append(job)
                    print(f"tank {str(winning_bid.tank.unique_id)} has won bid for {str(job)}")
                else:
                    print("job already taken")
                    
            for job in jobs_taken:
                self.cleaning_tank_jobs_list.discard(job)
                    
                
                    
                
    def step(self):
        """
        Keep running until all pieces of plastic are collected
        """
        self.tick = self.tick + 1
        object_is_plastic = lambda a : True if isinstance(a,LWPlastic) or isinstance(a, HeavyPlastic) or isinstance(a, CompactedPlastic) else False
        
        plastic = [a for a in self.schedule.agents if object_is_plastic(a)]
        remaining_plastic = [x for x in plastic if not x.picked]
        
        if len(remaining_plastic) > 0:
#! new code 
            #spawns in plastic at a random rate
            if self.added_functionality:
                if self.schedule.steps % self.heavy_plastic_current_spwan_rate == 0: 
                    #randomly change the spwan rate of plastic 
                    self.heavy_plastic_current_spwan_rate = random.choice(np.arange(self.heavy_plastic_maximum_spwan_rate, self.heavy_plastic_minimum_spwan_rate))
                    print("New heavy plastic spawn rate is "+str(self.heavy_plastic_current_spwan_rate))
                    while True: 
                        cell = random.choice(np.arange(self.width)), random.choice(np.arange(self.height))
                        if self.grid.is_cell_empty(cell):
                            break
                    
                    heavy_plastic = HeavyPlastic(self.last_id, self, cell)
                    self.last_id += 1
                    self.schedule.add(heavy_plastic)
                    self.grid.place_agent(heavy_plastic, cell)
                    print("New heavy plastic peice spwaned at "+str(cell))
                    
                        
            self.schedule.step()
#!added the below
            #auctions jobs
            if self.added_functionality:
                self.auction_jobs()
                
            print("Cleaning tank plastic tracket", end=" - ")
            print(self.cleaning_tank_plastic_tracker)
            print("Compact plastic locations", end=" - ")
            print(self.compacted_plastic_locations)
            
            if self.rubbish_collection_robots: 
                self.assign_rubbish_robots_jobs()
            print("-------------------------------------------------------------------------------------------------------")
            
        else:
            print("Finished")
            self.running = False
        print("collected")    
        self.datacollector.collect(self)
            
        
        