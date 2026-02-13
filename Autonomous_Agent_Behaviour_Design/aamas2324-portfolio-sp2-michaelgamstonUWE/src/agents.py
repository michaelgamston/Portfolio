import mesa
from mesa.model import Model
import numpy as np 
import math
import random



NUMBER_OF_CELLS = 50
SEARCHING = 1
FINISHED = 2 
FULL = 3
MOVING = 4
AVOIDANCE = 5
BATTERY_EMPTY = 6
READY = 7 

class RobotParent(mesa.Agent):
    """Parent class that all robot tpyes inherit from 
    contains boiler plate code that can be overwritten if a robot class
    requires customisation 

    Args:
        mesa (Agent): the main mesa super class
    """
    
    def __init__(self, unique_id : int, model : mesa.Model, pos : tuple, speed : int, state : int, target : tuple, bin_loc : tuple) -> None:
        super().__init__(unique_id, model)
        self.priority = unique_id+1
        self.x, self.y = pos
        self.next_x, self.next_y = None, None
        self.is_stuck = False
        self.hopper_fullness = 0
        self.speed = speed
        self.state = state
        self.target = target
        self.bin_loc = bin_loc
        self.avoidance_step_tracking = []
        
    def compact_and_drop_plastic(self) -> None: 
        """When the rubbish collection robots are selected, 
        this function is called instedad of robots going to the bin to empty
        """
        self.model.add_compacted_plastic((self.x , self.y))
        self.empty_hopper()
    
        
    def find_path_to_target(self) -> None:
        """
            If state is MOVING 
            Find the next position to move to towards the next search postion 
        
        """
        #find direction to the correct start posistion
        if self.y != self.target[1]:
            self.next_y = self.y+1 if self.target[1] > self.y else self.y-1  
        else: 
            self.next_y = self.y
            
        if self.x != self.target[0]:
            self.next_x = self.x+1 if self.target[0] > self.x else self.x-1
        else: 
            self.next_x = self.x
        
    def is_object_immovable(self, object) -> bool: 
        """checks object type 

        Args:
            object (_type_): object to be assessed

        Returns:
            bool: True if object is immovable (Tree, Bin, Charger) 
        """
        if isinstance(object, PalmTree):
            return True
        elif isinstance(object, Bin):
            return True 
        elif isinstance(object, HeavyPlastic):
            return True
        elif isinstance(object, ChargingStation):
            return True
        elif isinstance(object, CompactedPlastic):
            return True
        
        return False
        
    def is_object_a_robot(self, object) -> bool:
        """checks object type

        Args:
            object (_type_): object to be assessed

        Returns:
            bool: True is object is a robot (cleaing tank or light robot)
        """
        
        if isinstance(object, LWRobot):
            return True 
        if isinstance(object, CleaningTank):
            return True    
        if isinstance(object, RubishRobot):
            return True
        return False 
        
    def i_am_stuck(self) -> None:
        """setter for if the robot is stuck
        """
        self.is_stuck = True
        self.priority = (self.unique_id+1)*1000

   
    def i_am_unstuck(self) -> None:
        """returns robot back to unstuck 
        """
        self.is_stuck = False
        self.priority = (self.unique_id+1)
        
    #Overwrite
    def i_am_full(self) -> None:
        """setter for if the robot is full
        """
        pass
    
    #Overwrite    
    def empty_hopper(self) -> None:
        """empties the hopper
        """
        pass


    def is_cell_inrange(self, cell) -> bool:
        """checks a cell is inrange 
        stops robots going from one side to the other

        Args:
            cell (tuple): cooridate

        Returns:
            bool: True if in range
        """
        x_range = np.arange(self.x-1, self.x+2)
        y_range = np.arange(self.y-1, self.y+2)
  
        if cell[0] in x_range:
            if cell[1] in y_range:
                return True
            
        return False
    
    
    #Overwrite 
    def avoid_static_object(self, object) -> bool:
        """This alogrithm find an alternative square to move based on a weighted 
        euclidean distance

        Args:
            object (_type_): object that is in current path

        Returns:
            bool: True if successful
        """
        pass
    
    
    #Overwrite 
    def check_collision(self) -> str:
        """check for collisions 

        Returns:
            str: a function call
        """ 
        pass
    
    
    #Overwrite 
    def change_state(self, state :int) -> None:
        """changes the state of the robot 
        also does some setter machanic for certain states

        Args:
            state (int): the state we wan to change to
        """
        pass
    
    
    #Overwrite 
    def decision(self) -> str:
        """make a decision on what the robot should do 

        Returns:
            str: a function for step to call 
        """
        pass
    
    
    #Overwrite
    def wait(self) -> None:
        """keeps the robot in the same position
        """
        self.next_x, self.next_y = self.x, self.y 
        
    #Overwrite
    def pick(self) -> None:
        """picks and deletes a peice of plastic from the robots current location
        """
        pass
       
    #Overwrite if needed
    def move(self) -> None:
        """moves' the robot
            includes a saftey check incase assignment of one of the axis failed
        """   
        self.model.grid.move_agent(self, (self.next_x, self.next_y))
    
    #Overwrite 
    def step(self) -> None:
        """readies the robot for next action 
        """
        pass
    
    #Overwrite 
    def advance(self) -> None:
        """advances the robots x and y, prints end of turn information
        """
        self.x, self.y = self.next_x, self.next_y 
        
        print(f"{str(type(self))} {str(self.unique_id)}'s current position is {str((self.x, self.y))}", end = " - ")
        print(f"target position {str(self.target)}", end = " - ")
        print(f"Current state {str(self.state)}", end= " - ")
        print(f"priority is {str(self.priority)}", end = " - ")
        print(f"speed is {str(self.speed)}", end = "  -  ")
        print(str(self.hopper_fullness))
        
        
    def print_finish_message(self) -> None:
        """prints a message when the robot has completed it search 
        """
        print(f"Robot {str(self.unique_id)} has finished searching")


class LWRobot(RobotParent):
    """Represents a Box in the warehouse."""
    
    
    def __init__(self, unique_id : int, model : mesa.Model, pos : tuple, search_coordinates : list, bin_loc : tuple) -> None:
        """
        Intialise state and position of the box
        """
        #unique_id : int, model : mesa.Model, pos : tuple, speed : int, state : int, target : tuple, bin_loc : tuple
        super().__init__(unique_id=unique_id, model=model, pos=pos, speed=1, state=MOVING, target=search_coordinates.pop(0), bin_loc=bin_loc)
        print(f"Robot {str(unique_id)} is at start point {str(pos)}")
        self.search_coordinates = search_coordinates
        self.stuck_count = 0

        
    def i_am_full(self) -> None:
        """setter for if the robot is full
        """
        self.search_coordinates.insert(0, self.target)
        self.target = self.bin_loc
        self.priority = (self.unique_id+1)*100
        
        
     
    def empty_hopper(self) -> None:
        """empties the hopper
        """
        self.hopper_fullness = 0
        self.priority = self.unique_id+1
        self.speed = 1
    
        
    def get_next_search_pos(self) -> None: 
        """
            If state is SEARCHING
            Find the next search position to move to 
        """
        if contents_list := self.model.grid.get_cell_list_contents(self.target):
            for element in contents_list: 
                if isinstance(element, LWPlastic):
                    self.search_coordinates.append(self.target)
        try:
            self.target = self.search_coordinates.pop(0)
        except:
            self.change_state(FINISHED)
            self.next_x, self.next_y = (self.x, self.y)
            
    def update_cleaning_tank_jobs(self, pos : tuple) -> None: 
        """updates the cleaning tank job tracker

        Args:
            pos (tuple): the posistion of the job
        """
        if (pos not in self.model.cleaning_tank_jobs_list) or (pos not in self.model.cleaning_tank_plastic_tracker):
            self.model.cleaning_tank_jobs_list.add(pos)
            print("new heavy plastic job added at pos "+str(pos))
       
        
        
    def avoid_static_object(self, object) -> bool:
        """This alogrithm find an alternative square to move based on a weighted 
        euclidean distance

        Args:
            object (_type_): object that is in current path

        Returns:
            bool: True if successful
        """
        
        if self.model.added_functionality:
            if isinstance(object, HeavyPlastic):
                self.update_cleaning_tank_jobs((object.x, object.y))
            
        #check if our target is blocked by something immovable 
        if ((object.x, object.y) == self.target) and self.is_object_immovable(object):
            self.get_next_search_pos()
         
        if self.is_object_a_robot(object):
            if (object.x, object.y) == self.target: # to avoid death loops
                #if it's our last cell then we wait for it to be clear 
                #add block search pos to the back of the que then append the finishing space (None)
                if len(self.search_coordinates) == 0:
                    return False
                self.search_coordinates.append(self.target)
                self.get_next_search_pos()
                
                
                
        
           
        self.change_state(AVOIDANCE)
        best_choice = 1000 #set overly high benchmark to start comparison
        proposed_cell = None

        for cell in self.model.grid.get_neighborhood((self.x, self.y), moore=True):

            if self.is_cell_inrange(cell):
                cell_contents_list = self.model.grid.get_cell_list_contents(cell)
                assess = True
                if cell_contents_list:
                    cell_contents = cell_contents_list[0]
                    if self.is_object_a_robot(cell_contents) or self.is_object_immovable(cell_contents):
                        assess = False
                
                if assess: 
                    #euclidean distance plus the number of times that cell has been choosen already when in advoidance
                    cell_distance_from_target = (math.dist(cell, self.target) + self.avoidance_step_tracking.count(cell))
                    if cell_distance_from_target < best_choice:
                        best_choice = cell_distance_from_target
                        proposed_cell = cell 
         
        if proposed_cell:
            self.next_x, self.next_y = proposed_cell
            return True           
        else: 
            return False
            
          
                                                                                               
    def check_collision(self) -> str: 
        """check for collisions 

        Returns:
            str: a function call
        """       
        
        if self.state == FINISHED: 
            return 'print_finish_message'
        
      
        if len(neighbors := self.model.grid.get_neighbors((self.x, self.y), moore = True, include_center = False, radius = 3)) > 0:
            blocker = False
            
                            
            #find if we have anything blocking our path
            blocker = self.model.grid.get_cell_list_contents((self.next_x, self.next_y))
            if blocker:
                blocker = blocker.pop(0)
                if isinstance(blocker, LWPlastic):
                    blocker = False
                

            # move if you have proirity    
            if self.priority < min([neighbor.priority for neighbor in neighbors]):
                if blocker:
                    if self.avoid_static_object(blocker):
                        return 'move'
                    else:
                        self.i_am_stuck()
                        return 'wait'
                else:
                    if self.state == AVOIDANCE:
                        self.exit_avoidance_state()
                    return 'move'  
            else:
                return 'wait'
        else: 
            if self.state == AVOIDANCE:
                self.exit_avoidance_state()
            return 'move'
         
           
    def exit_avoidance_state(self) -> None:
        """returns the robot to the state it was in before avoidance
        """
        
        if (self.hopper_fullness > 74):
            self.change_state(FULL)
        elif (self.x, self.y) == self.target:  
            self.change_state(SEARCHING)
        else: 
            self.change_state(MOVING)         

    
    def check_state(self) -> None:
        """check which state the robot needs to be in after every run 
        """

        if self.state == FINISHED:
            self.print_finish_message()
        elif self.hopper_fullness > 74:
            if self.model.rubbish_collection_robots:
                self.compact_and_drop_plastic()
            else:
                self.change_state(FULL) 
        elif (self.hopper_fullness > 69):
            self.speed = 2
        elif (self.next_x, self.next_y) == self.target:
            self.change_state(SEARCHING)  
             
        elif self.state == AVOIDANCE:
            pass
        
        else:
            self.change_state(MOVING)           
            
        self.search_coordinates = list(dict.fromkeys(self.search_coordinates))
            
                  
    def change_state(self, state :int) -> None:
        """changes the state of the robot 
        also does some setter machanic for certain states

        Args:
            state (int): the state we wan to change to
        """
         
        if state == FINISHED:

            #change priority to mimic an immovable object 
            self.priority = 100000
        elif state == AVOIDANCE:
            self.avoidance_step_tracking.append((self.x,self.y))

        elif state == FULL:
            if self.state != FULL: 
                self.i_am_full()
        else:
            self.avoidance_step_tracking.clear()
            
        self.state = state
        
        
    def decision(self) -> str:
        """make a decision on what the robot should do 

        Returns:
            str: a function for step to call 
        """
        if self.state == FINISHED:
            return 'print_finish_message'
        
        elif self.state == FULL: 
            self.find_path_to_target()
            if (self.next_x, self.next_y) == self.bin_loc:
                self.empty_hopper()
                  
        elif (self.state == MOVING) or (self.state == AVOIDANCE): 
            self.find_path_to_target()

        elif self.state == SEARCHING: 
            self.check_for_plastic()
            self.get_next_search_pos()
            
        return self.check_collision()
        
        
    def check_for_plastic(self) -> None:
        """check the current position for plastic
        """
        object = self.model.grid.get_cell_list_contents((self.x, self.y))
        if object: 
            object = object[0]
            if isinstance(object, LWPlastic):
                self.pick(object)
    
    
    def pick(self, plastic) -> None: 
        """picks up and deletes plastic from the grid

        Args:
            plastic (mesa.Agent): 
        """
        plastic.picked = True
        self.model.grid.remove_agent(plastic)
        self.hopper_fullness += 1
            
    
    def move(self) -> None:
        """moves' the robot
            includes a saftey check incase assignment of one of the axis failed
        """ 
        print("I actually have moved")       
        self.stuck_count = 0
        if self.is_stuck: 
            self.i_am_unstuck()
        self.model.grid.move_agent(self, (self.next_x, self.next_y))
        
    def wait(self) -> None:
        """keeps the robot in the same position
        """
        print("I'm actually waiting")
        self.stuck_count +=1
        self.next_x, self.next_y = self.x, self.y 
        
    
    def step(self) -> None:
        """readies the robot for next action 
        """
        print("Robot "+str(self.unique_id))
        print(f"Steps = {self.model.schedule.steps}, speed = {str(self.speed)}")
        if (mod := self.model.schedule.steps % self.speed) == 0: 
            print(f"Moved with {mod}")
            
            action = getattr(self, self.decision()) 
            action()
        else: 
            
            print(f"Waited with {mod}")
            self.wait()
            
        if self.hopper_fullness > 69: 
            print(self.search_coordinates)
        self.check_state() 
        
        
    
class CleaningTank(RobotParent):
    
    
    def __init__(self, unique_id : int, model : mesa.Model, pos : tuple, charger_loc : tuple, bin_loc : tuple, search_area_size : tuple) -> None:
        """
        Intialise state and position of the box
        """
        super().__init__(unique_id=unique_id, model=model, pos=pos, speed=2, state=MOVING, target=pos, bin_loc=bin_loc)
        print(f"Cleaing Tank {str(unique_id)} is at start point {str(pos)}")
        self.search_area_size = search_area_size
        #add one to avoid a robot getting priority 0 (which isn't easily manipulatable)
        self.stuck_step_tracker = 0
        self.charge = random.choice(np.arange(100,201))
        self.charger_loc = charger_loc

        
    
    def i_am_stuck(self) -> None:
        """setter for if the robot is stuck
        """
        self.stuck_step_tracker += 1
        self.is_stuck = True
        self.priority = (self.unique_id+1)*1000

   
    def i_am_unstuck(self) -> None:
        """returns robot back to unstuck 
        """
        self.stuck_step_tracker = 0
        self.is_stuck = False
        self.priority = self.unique_id+1
        
    
    def i_am_full(self) -> None:
        """setter for if the robot is full
        """
        self.target = self.bin_loc
        if self.model.added_functionality: 
            self.return_target_to_tracker(self.target)
        self.update_cloud_plastic_tracker(("Bin", "Bin"))
        self.priority = (self.unique_id+1)*100
        
        
    def empty_hopper(self) -> None:
        """empties the hopper
        """
        self.hopper_fullness = 0
        self.priority = self.unique_id+1
        

    def battery_emtpy(self) -> None: 
        """setter for if battery is empty
        """
        self.return_target_to_tracker(self.target)
        self.update_cloud_plastic_tracker(("Charger", "Charger"))
        self.target = self.charger_loc
        self.priority = (self.unique_id+1)*100
        
        
    def charge_battery(self) -> None:
        """returns charge back to 100
        """
        self.charge = 200   
        self.priority = self.unique_id+1
        self.change_state(MOVING)
        
        
    def change_state(self, state :int) -> None:
        """changes the state of the robot 
        also does some setter machanic for certain states

        Args:
            state (int): the state we wan to change to
        """
         
        
        if state == AVOIDANCE:
            self.avoidance_step_tracking.append((self.x,self.y))

        else:
            self.avoidance_step_tracking.clear()
            
        self.state = state
            
        
    def is_object_immovable(self, object) -> bool: 
        """checks object type 

        Args:
            object (_type_): object to be assessed

        Returns:
            bool: True if object is immovable (Tree, Bin, Charger) 
        """
        if isinstance(object, PalmTree):
            return True
        elif isinstance(object, Bin):
            return True 
        elif isinstance(object, ChargingStation):
            return True
        elif isinstance(object, CompactedPlastic):
            return True
        
        return False
    
    
    
    def avoid_static_object(self, object) -> bool:
        """This alogrithm find an alternative square to move based on a weighted 
        euclidean distance

        Args:
            object (_type_): object that is in current path

        Returns:
            bool: True if successful
        """
        if self.state == MOVING: 
            if (object.x, object.y) == self.target:
                self.target = self.find_direction()
                
      
        self.change_state(AVOIDANCE)
        best_choice = math.dist((self.x,self.y), self.target)*2 #set overly high benchmark to start comparison
        proposed_cell = None

 
        for cell in self.model.grid.get_neighborhood((self.x, self.y), moore=True):

            if self.is_cell_inrange(cell):
                cell_contents_list = self.model.grid.get_cell_list_contents(cell)

                assess = True
                if cell_contents_list:

                    cell_contents = cell_contents_list[0]
                    
                    if self.is_object_a_robot(cell_contents) or self.is_object_immovable(cell_contents):
                        assess = False
                
                if assess: 
                    
                    #euclidean distance plus the number of times that cell has been choosen already when in advoidance
                    cell_distance_from_target = (math.dist(cell, self.target) + self.avoidance_step_tracking.count(cell))

                    if cell_distance_from_target < best_choice:
                        best_choice = cell_distance_from_target
                        proposed_cell = cell 
         
        if proposed_cell:
            self.next_x, self.next_y = proposed_cell
            return True           
        else: 
            return False
    
    
    def check_collision(self) -> str: 
        """check for collisions 

        Returns:
            str: a function call
        """       
        
        
        if len(neighbors := self.model.grid.get_neighbors((self.x, self.y), moore = True, include_center = False, radius = 3)) > 0:
            blocker = False
            
                            
            #find if we have anything blocking our path
            blocker = self.model.grid.get_cell_list_contents((self.next_x, self.next_y))
            if blocker:
                blocker = blocker.pop(0)
                if isinstance(blocker, LWPlastic) or isinstance(blocker, HeavyPlastic):
                    blocker = False
                

            # move if you have proirity    
            if self.priority < min([neighbor.priority for neighbor in neighbors]):
                if blocker:
                    if self.avoid_static_object(blocker):
                        return 'move'
                    else:
                        self.i_am_stuck()
                        return 'wait'
                else:
                    if self.state == AVOIDANCE:
                        self.change_state(MOVING)
                    return 'move'  
            else:
                return 'wait'
        else: 
            if self.state == AVOIDANCE:
                self.change_state(MOVING)
            return 'move'
    
     
    def find_direction(self) -> tuple:
        """randomly picks target for cleaning tank

        Returns:
            tuple: _description_
        """
        while True:
            
            new_target = (np.random.choice(np.arange(self.search_area_size[0])), np.random.choice(np.arange(self.search_area_size[1])))
            contents = self.model.grid.get_cell_list_contents(new_target)
            if not contents:
                break
            else: 
                agent = contents[0]
                if (not self.is_object_immovable(agent)) and (not self.is_object_a_robot(agent)):
                    break 
        
        return new_target
    
    
    
    def return_target_to_tracker(self, target : tuple) -> None: 
        """Returns the given target to the job list in the case that it gets released

        Args:
            target (x,y): the posistion of the target
        """
        self.model.cleaning_tank_jobs_list.add(target)
    
    
    def update_cloud_plastic_tracker(self, pos) -> None:
        """updates the grid with the position of the plastic 
        that has been targeted 

        Args:
            pos (tuple): location of plastic 
        """
        self.model.cleaning_tank_plastic_tracker[self.unique_id-self.model.n_LWRobots] = pos
       
        
    def check_cloud_plastic_tracker(self, pos : tuple) -> bool: 
        """checks the cloud plastic tracker to see if job has already been uploaded

        Args:
            pos (tuple): position of job

        Returns:
            bool
        """
        if (pos in self.model.cleaning_tank_plastic_tracker) or (pos in self.model.cleaning_tank_jobs_list):
            return False
        else: 
            return True
                
    def scan_for_plastic(self) -> None: 
        """searches the surrounding 8 squares for plastic useing 'machine vision'
        """
        
        neighbors = self.model.grid.get_neighbors((self.x, self.y), moore=True, include_center = False, radius = 8)
        for neighbor in neighbors:
            if isinstance(neighbor, HeavyPlastic):
                if self.check_cloud_plastic_tracker((neighbor.x, neighbor.y)):
                    self.state = SEARCHING
                    self.target = neighbor.x, neighbor.y
                    self.update_cloud_plastic_tracker(self.target)

                
    def decision(self) -> None: 
        """make a decision on what the robot should do 

        Returns:
            str: a function for step to call 
        """
        
        if self.state == MOVING:
            self.scan_for_plastic()
        
        if(self.x, self.y) == self.target:
            if self.state == SEARCHING:
                self.pick()
            
            self.target = self.find_direction()
            self.update_cloud_plastic_tracker(self.target)
            
        if self.stuck_step_tracker > 10:
            self.target = self.find_direction()
            self.update_cloud_plastic_tracker(self.target)
                    
        self.find_path_to_target()
        
        if self.state == BATTERY_EMPTY or self.state == FULL:
              
            if (self.next_x, self.next_y) == self.target:
                if self.state == BATTERY_EMPTY:
                    self.charge_battery()
                    self.find_path_to_target()
                    
                elif self.state == FULL:
                    self.empty_hopper()
                    self.find_path_to_target()
          
           
        return self.check_collision()
     
    def pick(self) -> None:
        """picks and deletes a peice of plastic from the robots current location
        """
        self.state = MOVING
        object = self.model.grid.get_cell_list_contents((self.x, self.y))
        if object: 
            object = object[0]
            if isinstance(object, HeavyPlastic):
                object.picked = True
                self.model.grid.remove_agent(object)
                self.hopper_fullness +=1
                self.update_cloud_plastic_tracker(())

                   
    def move(self) -> None:
        """moves the robot and decreases it's charge
        """
        if self.model.added_functionality:
            self.charge -=1
        self.model.grid.move_agent(self, (self.next_x, self.next_y))
    
    
    def wait(self) -> None: 
        """sets the robot to wait in the same position 
        """
        self.next_x, self.next_y = self.x, self.y
        
        
    def bid(self, target : tuple ) -> int:
        """Function generates a bid based on fullness, battery life
        and distance to target

        Args:
            target (x,y): posistion of plastic 

        Returns:
            int: bid score 
        """
        #robots that are full won't be apart of the auction 
        
        bid = 0 
        
        if not self.model.rubbish_collection_robots:
            bid = 5 - self.hopper_fullness 
        distance_to_target = math.dist((self.x, self.y), target)
        
        #it's very important the robot has enough battery other wise the job will be reject and re-auctioned 
        #when the robot runs out 
        if self.charge > (distance_to_target + math.dist(target, self.charger_loc)):
            bid += 5
        else: 
            bid -= 5
            
        if distance_to_target < 5: 
            bid += 5
        elif distance_to_target < 10:
            bid += 4
        elif distance_to_target < 20: 
            bid += 3
        elif distance_to_target < 40: 
            bid += 2
        elif distance_to_target < 80: 
            bid += 1
            
        return bid
    

    def assign_new_job(self, target : tuple) -> None:
        """assigns a new job once robot has won a bid

        Args:
            target (tuple): posistion of plastic
        """
        self.target = target 
        self.state = SEARCHING
        self.update_cloud_plastic_tracker(target)
        
        
        
    def step(self) -> None:
        """sets the speed and checks for fullness 
        the calls decision 
        """

        if self.model.schedule.steps % self.speed == 0: 
            if self.model.added_functionality:    
                if self.state != BATTERY_EMPTY:
                    distance_to_battery = round(math.dist((self.x, self.y), self.charger_loc)) 
                    if distance_to_battery >= self.charge:
                        self.battery_emtpy()
                        self.change_state(BATTERY_EMPTY)
                        
            if self.hopper_fullness > 4:
                if self.model.rubbish_collection_robots:
                    self.model.add_compacted_plastic((self.x , self.y))
                    self.empty_hopper()
                else:
                    self.i_am_full()
                    self.change_state(FULL)
                   
            action = getattr(self, self.decision()) 
            action()

        else:
            self.wait()
    
    
    
class RubishRobot(RobotParent):
    
    def __init__(self, unique_id : int, model : mesa.Model, pos : tuple, bin_loc : tuple) -> None:
        """
        Intialise state and position of the box
        """
        super().__init__(unique_id=unique_id, model=model, pos=pos, speed=1, state=READY, target=(None, None), bin_loc=bin_loc)
        print(f"Rubbish Robot {str(unique_id)} is at start point {str(pos)}")
        
        
    def is_object_immovable(self, object) -> bool: 
        """checks object type 

        Args:
            object (_type_): object to be assessed

        Returns:
            bool: True if object is immovable (Tree, Bin, Charger) 
        """
        if isinstance(object, PalmTree):
            return True
        elif isinstance(object, HeavyPlastic):
            return True
        elif isinstance(object, ChargingStation):
            return True
        
        
    def is_object_a_robot(self, object) -> bool:
        """checks object type

        Args:
            object (_type_): object to be assessed

        Returns:
            bool: True is object is a robot (cleaing tank or light robot)
        """
        
        if isinstance(object, LWRobot):
            return True 
        if isinstance(object, CleaningTank):
            return True    
        return False 
        
        
    def assign_new_job(self, job : tuple) -> None: 
        """assigns the rubbish robot a new job

        Args:
            job (tuple): the location of the job 
        """
        self.state = SEARCHING
        self.target = job
        self.priority = self.unique_id+1
        
        
    def pick(self) -> None: 
        object_list = self.model.grid.get_cell_list_contents((self.x, self.y))
        if object_list:
            for object in object_list:  
                if isinstance(object, CompactedPlastic):
                    object.picked = True
                    self.model.grid.remove_agent(object)
                    return
            
        print(f"Rubish Robot {self.unique_id} - No Compact Plastic found - Pick Failed")
        self.model.compacted_plastic_locations.append((object.x, object.y))
                
        
    def avoid_static_object(self, object) -> bool:
        """This alogrithm find an alternative square to move based on a weighted 
        euclidean distance

        Args:
            object (_type_): object that is in current path

        Returns:
            bool: True if successful
        """
        
        self.change_state(AVOIDANCE)
        best_choice = math.dist((self.x,self.y), self.target)*2 #set overly high benchmark to start comparison
        proposed_cell = None

 
        for cell in self.model.grid.get_neighborhood((self.x, self.y), moore=True):

            if self.is_cell_inrange(cell):
                cell_contents_list = self.model.grid.get_cell_list_contents(cell)

                assess = True
                if cell_contents_list:

                    cell_contents = cell_contents_list[0]
                    
                    if self.is_object_a_robot(cell_contents) or self.is_object_immovable(cell_contents):
                        assess = False
                
                if assess: 
                    
                    #euclidean distance plus the number of times that cell has been choosen already when in advoidance
                    cell_distance_from_target = (math.dist(cell, self.target) + self.avoidance_step_tracking.count(cell))

                    if cell_distance_from_target < best_choice:
                        best_choice = cell_distance_from_target
                        proposed_cell = cell 
         
        if proposed_cell:
            self.next_x, self.next_y = proposed_cell
            return True           
        else: 
            return False
            
     
    def change_state(self, state :int) -> None:
        """changes the state of the robot 
        also does some setter machanic for certain states

        Args:
            state (int): the state we wan to change to
        """
        if state == AVOIDANCE:
            self.avoidance_step_tracking.append((self.x,self.y))

        else:
            self.avoidance_step_tracking.clear()
            
        self.state = state
        
    
    def exit_avoidance_state(self) -> None:
        """returns the robot to the state it was in before avoidance
        """
        
        if self.target == self.bin_loc: 
            self.change_state(MOVING)
        else: 
            self.change_state(SEARCHING)
    
    
    def check_collision(self) -> str:
        """check for collisions 

        Returns:
            str: a function call
        """       
        
      
        if len(neighbors := self.model.grid.get_neighbors((self.x, self.y), moore = True, include_center = False, radius = 3)) > 0:
            blocker = False
            
                            
            #find if we have anything blocking our path
            blocker = self.model.grid.get_cell_list_contents((self.next_x, self.next_y))
            if blocker:
                blocker = blocker.pop(0)
                if isinstance(blocker, LWPlastic) or isinstance(blocker, CompactedPlastic) or isinstance(blocker, Bin):
                    blocker = False
              
            #if robots are in the bin we want to ignore them  
            for index, neighbor in enumerate(neighbors):
                if isinstance(neighbor, RubishRobot):
                    if neighbor.state == READY: 
                        neighbors.pop(index)
                        
                        
            # move if you have proirity    
            if self.priority < min([neighbor.priority for neighbor in neighbors]):
                if blocker:
                    if self.avoid_static_object(blocker):
                        return 'move'
                    else:
                        self.i_am_stuck()
                        return 'wait'
                else:
                    if self.state == AVOIDANCE:
                        self.exit_avoidance_state()
                    return 'move'  
            else:
                return 'wait'
        else: 
            if self.state == AVOIDANCE:
                self.exit_avoidance_state()
            return 'move'
    
    
    def step(self) -> None:
        """steps the robot
        """
        if (self.x, self.y) == self.target:
            
            if self.state == SEARCHING:
                self.pick()
                self.target = self.bin_loc
                self.state = MOVING
                
            elif self.state == MOVING: 
                self.taget = (None, None)
                self.state = READY
                
        if self.state == READY: 
            self.priority = 100000
            self.wait()
        else:
            self.find_path_to_target()
            action = getattr(self, self.check_collision()) 
            action()
            
                   
        
            
                    
class CompactedPlastic(mesa.Agent):
    
    def __init__(self, unique_id: int, model: Model, pos) -> None:
        super().__init__(unique_id, model)
        self.priority = 100000
        self.x, self.y = pos
        self.picked = False
    
    
class PalmTree(mesa.Agent):
    
    def __init__(self, unique_id: int, model: Model, pos : tuple) -> None:
        super().__init__(unique_id, model)
        #all immovable agents have priority 100000
        self.priority = 100000
        self.x, self.y = pos
        
        
class LWPlastic(mesa.Agent):
    
    def __init__(self, unique_id: int, model: Model, pos : tuple) -> None:
        super().__init__(unique_id, model)
        #all immovable agents have priority 100000
        self.priority = 100000
        self.x, self.y = pos
        self.picked = False
        
        
class HeavyPlastic(mesa.Agent):
    def __init__(self, unique_id: int, model: Model, pos : tuple) -> None:
        super().__init__(unique_id, model)
        #all immovable agents have priority 100000
        self.priority = 100000
        self.x, self.y = pos
        self.picked = False


class Bin(mesa.Agent):
    
    def __init__(self, unique_id: int, model: Model, pos : tuple) -> None:
        super().__init__(unique_id, model)
        #all immovable agents have priority 100000
        self.priority = 100000
        self.x, self.y = pos
        
        
    
        
class ChargingStation(mesa.Agent):
    
    def __init__(self, unique_id: int, model: Model, pos : tuple) -> None:
        super().__init__(unique_id, model)
        #all immovable agents have priority 100000
        self.priority = 100000
        self.x, self.y = pos
        