import numpy as np
import statistics
import networkx as nx
import matplotlib.pyplot as plt
import logging 
import os
import shutil
import time
import sys
import json

class ACO_TNDP():
    
    def __init__(self, custom_logging_filepath:str = None, custom_data_path:str = None, k:int = 10, e:int = 0.8, q:int = 100, a:int = 3, b:int = 1, y:int = 2, x:int = 1,
                 num_routes:int = 12, max_length:int = 15, stop:int = 3, folds:int = 3, locations_provided:bool = False, recombination:bool = False, provide_graphs = False) -> None:
                                                                                       
        try:
            assert(type(custom_logging_filepath) == str or custom_logging_filepath == None)
        except (AssertionError, ) as e: 
            print("------ error thrown ---------")
            print(e)
            print("Please enter a null value or string for custom_logging_filepath")
            sys.exit()
           
        try:
            assert(type(folds) == int)
            assert(folds>0)
        except AssertionError as e: 
            print("------ error thrown ---------")
            print(e)
            print("Please enter a value for FOLDS that is greater than 0")
            sys.exit() 
              
        self.graph_dir = 'output/graphs/'
        self.provide_graphs = provide_graphs
            
        if os.path.isdir(self.graph_dir):
            shutil.rmtree(self.graph_dir)
            time.sleep(1)
        os.makedirs(self.graph_dir)
            
        for fold in range(folds): 
            if not os.path.isdir(self.graph_dir+f'fold_{fold}/'):
                os.makedirs(self.graph_dir+f'fold_{fold}/')
            
        if custom_logging_filepath:
            logger_file = custom_logging_filepath
        else:
            logger_file = f'ACO_TNDP_output.txt'

        #logger 
        logging.basicConfig(filename= logger_file, format='%(asctime)s %(message)s', filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        try:
            assert(type(locations_provided) == bool)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a boolean value for locations_provided")
            sys.exit()
            
        try:
            assert(type(recombination) == bool)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a boolean value for recombination")
            sys.exit()
            
            
        try:
            assert(type(provide_graphs) == bool)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a boolean value for provide_graphs")
            sys.exit()
            
        try:
            assert(type(custom_data_path) == str or custom_data_path == None)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a a null value or string for custom_data_path")
            sys.exit()
            
        try:
            assert(type(k) == int or type(k) == float)
            assert(k>=0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for K that is greater than or equal 0")
            sys.exit()
            
        try:
            assert(type(e) == int or type(e) == float)
            assert(e>=0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for E that is greater than or equal 0")
            sys.exit()
            
        try:
            assert(type(q) == int or type(q) == float)         
            assert(q>=0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for Q that is greater than or equal 0")
            sys.exit()
                       
        try:    
            assert(type(a) == int or type(a) == float)
            assert(a>=0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for A that is greater than or equal 0")
            sys.exit()
            
        try:
            assert(type(b) == int or type(b) == float)
            assert(b>=0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for B that is greater than or equal 0")
            sys.exit()
            
        try:
            assert(type(y) == int or type(y) == float)
            assert(y>=0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for Y that is greater than or equal 0")
            sys.exit()
            
        try:
            assert(type(x) == int or type(x) == float)
            assert(x>=0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for X that is greater than or equal 0")
            sys.exit()
            
        try:
            assert(type(num_routes) == int)
            assert(num_routes>0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for NUM_ROUTES that is greater than 0")
            sys.exit()
            
        if max_length != None: 
            try:
                assert(type(max_length) == int)
                assert(max_length>0)
            except AssertionError as e: 
                print("------ error thrown check logs ---------")
                self.logger.info(e)
                self.logger.info("Please enter a value for MAX_LENGTH that is greater than 0")
                sys.exit()
                
        try:
            assert(type(stop) == int)
            assert(stop>0)
        except AssertionError as e: 
            print("------ error thrown check logs ---------")
            self.logger.info(e)
            self.logger.info("Please enter a value for STOP that is greater than 0")
            sys.exit()
            
        
        
        self.recombination = recombination
        
        self.locations_provided = locations_provided
        
        if custom_data_path:
            self.data_folder = custom_data_path
        else:
            self.data_folder = f"data/graph_data"
            
        if self.locations_provided:
            self.coords = f'{self.data_folder}/Coords.txt'
        self.demand = f'{self.data_folder}/Demand.txt'
        self.visibility = f'{self.data_folder}/Visibility.txt'

        self.K = k  # number of ants 
        self.E = e # evapoartion rate
        self.Q = q # pheromone laying rate 
        self.A = a # pheremone weight
        self.B = b # distance weight / route visibility 
        self.Y = y # edge busyness weight
        self.X = x
        self.EDGE_NOT_COVERED = None

        self.ROUTES = num_routes
        self.MAX_LENGTH = max_length
        self.STOP = stop
        self.FOLDS = folds

        self.AVOID_EDGES_THAT_ARE_COVERED = True
        
        self.ant_metadata = {}
        for ant in range(self.K):
            self.ant_metadata[ant] = {
                'start_node' : None, 
                'current_node' : None, 
                'path_tracker' : [],
                'fitness' : 0,
                'i_have_been_stuck_once_already' : False
            }   
        
        self.bus_stops_connected = []
        self.route_tracker = {}
        self.individual_edges = {}
        self.edge_tracker = []

        for route in range(self.ROUTES): 
            self.route_tracker[route] = []
            self.individual_edges[route] = []
            
        self.location_matrix = []
        self.all_nodes = []
        self.distance_matrix = []
        self.connected_nodes = {}
        self.normalised_distance_matrix = []
        self.pheromone_matrix = []
        self.busyness_matrix = []
        self.normalised_busyness_matrix = []
        self.mean_of_none_zero_distance_elements = 0
        
        
       
            

        
    def scale_features(self,features : list) -> list:
        """scales feature to between 1 and 101
        takes the max number and add's 1% padding to it.
        the features are then transcribed to be percentages of this number
        scaled features to between 1 and 101 as we need to avoid have anything > 1 for the selection aglorithm
        
        Args:
            features (list): feature list

        Returns:
            list: scaled feature list
        """
        try:
            max_value = max(features)
            
            padded_max = max_value + ((max_value/100)*1)
            
            normalised_data = []
            for x in features: 
                normalised_data.append(round((x/padded_max)*100, 2)+1)
        except TypeError as e: 
                    print("------ error thrown check logs ---------")
                    self.logger.warning(e)
                    self.logger.warning("Data scaler recieved unexpect vairbale type\n\
                        this could mean there are space's seperating the rows in your input data\n\
                        Please confirm input data is the correct format and type")
                    sys.exit()
                    
        return normalised_data
    
                   
    
    def load_graph_data(self):
        """loads in all the data from the inputs files
        """
        
        #Location
        if self.locations_provided:
            with open(self.coords, 'r') as f: 
                for line in f.readlines():
                    try:
                        assert(line != False)
                        coordinate = [float(x) for x in line.split()]
                        assert(len(coordinate) == 2)
                        self.location_matrix.append(coordinate)
                    except (AssertionError, TypeError, ValueError) as e: 
                        print("------ error thrown check logs ---------")
                        self.logger.warning(e)
                        self.logger.warning("Location matrix has invalid values in it this could mean - \n\
                                        the structure of the file is incorrect\n\
                                        some values are negative\n\
                                        some values are the incorrect data type\n\
                                        the sum of a nodes connections is 0 meaning it has no connections")
                        sys.exit()
          
        #Distance          
        with open(self.visibility, 'r') as f: 
            for line in f.readlines():
                row = []
                try: 
                    assert(line != False)
                    for x in line.split():
                        element = float(x) if x != 'Inf' else 0
                        assert(element >= 0)
                        row.append(element) 
                    
                
                    assert(sum(row) > 0)
                except (AssertionError, TypeError, ValueError) as e: 
                    print("------ error thrown check logs ---------")
                    self.logger.warning(e)
                    self.logger.warning("Visibility matrix has invalid values in it this could mean - \n\
                                    the structure of the file is incorrect\n\
                                    some values are negative\n\
                                    some values are the incorrect data type\n\
                                    the sum of a nodes connections is 0 meaning it has no connections")
                    sys.exit()
                        
                self.distance_matrix.append(row)  
            
        #Demand
        with open(self.demand, 'r') as f: 
            for line in f.readlines():
                row = []
                try:
                    assert(str(line) != None)
                    for x in line.split():
                        element = float(x)
                        assert(element >= 0)
                        row.append(element) 
                        
                 
                    assert(sum(row) >= 0)
                except (AssertionError, TypeError, ValueError) as e: 
                    print("------ error thrown check logs ---------")
                    self.logger.warning(e)
                    self.logger.warning("Demand matrix has invalid values in it this could mean - \n\
                                    some values are negative\n\
                                    the structure of the file is incorrect\n\
                                    some values are the incorrect data type")
                    sys.exit()
                    
                self.busyness_matrix.append(row)
                      
        try: 
            if self.locations_provided:
                assert(len(self.distance_matrix) == len(self.busyness_matrix))
                assert(len(self.distance_matrix) == len(self.location_matrix))
                assert(len(self.busyness_matrix) == len(self.location_matrix))
            else:
                assert(len(self.distance_matrix) == len(self.busyness_matrix))
                
            row_len = len(self.distance_matrix)
            for r in self.distance_matrix: 
                assert(len(r) == row_len)
                
            for r in self.busyness_matrix: 
                assert(len(r) == row_len)
            
        except (AssertionError, TypeError) as e: 
            print("------ error thrown check logs ---------")
            self.logger.warning(e) 
            self.logger.warning("distance, visibility and/ or location matrices are unequal sizes\nThis could mean their are unequal rows\nRows are unequal lengths") 
            sys.exit()       
            
        self.all_nodes = np.arange(len(self.distance_matrix))
        
        for index_r, row in enumerate(self.distance_matrix):
            node_to_node = []
            for index_e, element in enumerate(row):
                if element != 0: 
                    node_to_node.append(index_e)
            self.connected_nodes[index_r] = node_to_node
        
        self.normalised_distance_matrix = np.reshape(self.scale_features(np.array(self.distance_matrix).flatten()), np.shape(self.distance_matrix))
        #flip so we reward short distances in the selection equation
        for index_r, row in enumerate(self.normalised_distance_matrix):
            for index_e, element in enumerate(row):
                if element != 1: 
                    self.normalised_distance_matrix[index_r][index_e] = 102 - element 
                    
        all_row_means = []
        for row in self.normalised_distance_matrix:
            row_mean = []
            for element in row: 
                if element != 1: 
                    row_mean.append(element)
                    
            all_row_means.append(statistics.mean(row_mean))
                
        self.mean_of_none_zero_distance_elements = statistics.mean(all_row_means) 
        self.pheromone_matrix = np.ones(np.shape(self.distance_matrix))*self.mean_of_none_zero_distance_elements
            
        self.normalised_busyness_matrix = np.reshape(self.scale_features(np.array(self.busyness_matrix).flatten()), np.shape(self.busyness_matrix))
        
        edges = []
        for key in self.connected_nodes: 
            for node in self.connected_nodes[key]:
                edges.append((key, node))
                
        if self.provide_graphs:    
            self.draw_graph(edges, "Complete_Map", "black", False, False)
        
        
    
    def draw_graph(self, edge_list:list, title:str, color:str, save_location:str, route:list) -> None: 
        """draws a graph and saves it useing the given parameters

        Args:
            edge_list (list): _description_
            title (str): _description_
            color (str): _description_
            save_location (str): _description_
            route (list): _description_
        """
        plt.figure(figsize=(15,12))
        
        if route:
            plt.title(title+"\n"+str(route))
        else: 
            plt.title(title)
            
        G = nx.Graph()
        
        if self.locations_provided:
            for index, pos in enumerate(self.location_matrix):
                G.add_node(index, pos=pos)
                
            pos = nx.get_node_attributes(G, 'pos')  
        else: 
            for index in range(len(self.distance_matrix)):
                G.add_node(index)
            
        for edge in edge_list:
            G.add_edge(edge[0], edge[1], color=color)
    
          
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        
        if self.locations_provided:
            nx.draw(G, pos=pos, edge_color=colors, with_labels=True) 
        else:  
            nx.draw(G, edge_color=colors, with_labels=True) 
        
        if save_location: 
            plt.savefig(f'{save_location}{title}.png')
        else:
            plt.savefig(f'{self.graph_dir}{title}.png')
            
        plt.close()
        
    
        
    def find_start_node(self) -> int:
        """find the busyest node to start at
            if all nodes have been serviced a node is choosen at random

        Returns:
            int: start node
        """
        #the busyness is eqaul to the len of the busyness matrix if the node has been completely serviced
        busyness = len(self.normalised_busyness_matrix)
        start_node = None
        for node, list in enumerate(self.normalised_busyness_matrix):
            if sum(list) > busyness:
                busyness = sum(list)
                start_node = node
        
        if start_node == None: 
            start_node = np.random.choice(np.arange(0, len(self.normalised_busyness_matrix)))
            
        return start_node
    
    
    
    def pick_next_node(self, ant : int) -> int: 
        """computes a solution to the node selection algorithm 

        Args:
            ant (int): the ant that is asking 

        Returns:
            int: node to move to 
        """

        while True: 
            possible_nodes = [x for x in self.connected_nodes[self.ant_metadata[ant]['current_node']] if x not in self.ant_metadata[ant]['path_tracker']]
            
            if len(possible_nodes) > 0:
                break
            else: 
                if self.ant_metadata[ant]['i_have_been_stuck_once_already']:
                    return -1
                else:
                    self.ant_metadata[ant]['i_have_been_stuck_once_already'] = True
                    self.ant_metadata[ant]['current_node'] = self.ant_metadata[ant]['start_node']
                    self.ant_metadata[ant]['path_tracker'].reverse()

        current_node = self.ant_metadata[ant]['current_node'] 
        if self.AVOID_EDGES_THAT_ARE_COVERED:
            demoninator = sum([(self.normalised_distance_matrix[current_node][x]**self.B)*
                        (self.pheromone_matrix[current_node][x]**self.A)*
                        (self.normalised_busyness_matrix[current_node][x]**self.Y)*
                        (1 if ((current_node, x) in self.edge_tracker) or ((x, current_node) in self.edge_tracker) else self.mean_of_none_zero_distance_elements)**self.X for x in possible_nodes])
        else:
            demoninator = sum([(self.normalised_distance_matrix[current_node][x]**self.B)*
                        (self.pheromone_matrix[current_node][x]**self.A)*
                        (self.normalised_busyness_matrix[current_node][x]**self.Y) for x in possible_nodes])

        p_dict = {}
        for node in possible_nodes:
            distance = self.normalised_distance_matrix[current_node][node]
            pheromone = self.pheromone_matrix[current_node][node]
            busyness = self.normalised_busyness_matrix[current_node][node]
            
            if self.AVOID_EDGES_THAT_ARE_COVERED: 
                if ((current_node, node) in self.edge_tracker) or ((node, current_node) in self.edge_tracker):
                    repeated_edge_avoidance = 1
                else: 
                    repeated_edge_avoidance = self.mean_of_none_zero_distance_elements
                    
                p_dict[node] = ((distance**self.B)*(pheromone**self.A)*(busyness**self.Y)*(repeated_edge_avoidance**self.X)) / demoninator
            else: 
                p_dict[node] = ((distance**self.B)*(pheromone**self.A)*(busyness**self.Y)) / demoninator

        node_picker = np.random.random()
        sum_of_p = 0
        for key in p_dict.keys():
            sum_of_p += p_dict[key]
            if sum_of_p > node_picker:
                return key


    def do_these_lists_share_elements(self, list1:list, list2:list) -> dict: 
        """finds items in common between two list 
        returns the shared items, their postions and the two list

        Args:
            list1 (list): route
            list2 (list): route

        Returns:
            dict: shared_items
        """
        
        metadata = {'shared_items' : []}
        found_anything = False
        
        for index, item in enumerate(list1): 
            if item in list2: 
                metadata['shared_items'].append(item) 
                found_anything = True

        if found_anything:
            return metadata
        else: 
            return False
        
        
        
    def find_route_length(self, route:list) -> int: 
        """find the length of a given route

        Args:
            route (list): route

        Returns:
            int: length
        """

        length = 0
        
        for index in range(len(route[:-1])):
            length += self.distance_matrix[route[index]][route[index+1]]
            
        return length
    
    
    def find_route_using_transfer(self, origin:int, destination:int, route1:list, route2:list, shared_items:list) -> tuple:
        """fins the shorest route from origin to destination using a shared node

        Args:
            origin (node): start node
            destination (node): end node
            route1 (list): origin route
            route2 (list): destination route
            shared_items (list): list of shared nodes

        Returns:
            tuple: (combind route, fitness)
        """
        best_route = []
        grade = 1000000000000
        #print(f'finding route from {route1} to {route2} using shared nodes {shared_items}\nfrom origin {origin} to destination {destination}')
        for shared_item in shared_items:
            
            #figure out which side of the shared item origin is on and splice list
            if route1.index(origin) < route1.index(shared_item):
                spliced_route1 = route1[route1.index(origin): route1.index(shared_item)+1]
            else: 
                spliced_route1 = route1[route1.index(shared_item): route1.index(origin)+1]
                spliced_route1.reverse()
                
            
            #figure out which side of the shared item destination is on and splice list
            if route2.index(destination) < route2.index(shared_item):
                spliced_route2 = route2[route2.index(destination) : route2.index(shared_item)+1]
                spliced_route2.reverse()
            else:
                spliced_route2 = route2[route2.index(shared_item) : route2.index(destination)+1]
                
            appened_route = spliced_route1
            appened_route.extend(spliced_route2[1:])    
            
                
            route_length = self.find_route_length(appened_route)
            if route_length < grade:
                grade = route_length
                best_route = appened_route
                
        return (best_route, grade)



    def grade_solution(self, final_grading:bool = False):
        """Provides a value for the loss function by computing: 
        TT + (TTR * 5) + ((ATT * TU) + 50)

        Args:
            final_grading (bool, optional): If true then extra metrics are produced: 
            d0 percentage of demand serviced with no transfers 
            d1 percentage of demand serviced with one transfers 
            d2 percentage of demand serviced with two transfers 
            dun percentage of demand unsatisfied
            att average travel time 

        Returns:
            _type_: int or set(int)
        """

        if final_grading:
            d0 = 0
            d1 = 0
            d2 = 0
            dun = 0
            total_demand = 0
        
        tt = 0
        ttr = 0
        tu = 0
        travel_times = []
        nodes_to_assess = [x for x in range(len(self.busyness_matrix))]
        
        all_routes = [x for x in self.route_tracker.values() if x]
        total_connected_nodes = np.unique([x for sublist in all_routes for x in sublist])
        for index_d, row in enumerate(self.busyness_matrix):
            destination = index_d
            for node_o in nodes_to_assess:
                origin = node_o
                demand = int(row[origin])
                
                if final_grading: 
                    total_demand += demand
                    
                    
                if (origin in total_connected_nodes) and (destination in total_connected_nodes):
                    origin_routes = []
                    destination_routes = []
                    
                    for route in all_routes:
                        
                        #figure out which routes the origin is in 
                        if origin in route: 
                            origin_routes.append(route)
                            
                        #figure out which routes the destination is in   
                        if destination in route: 
                            destination_routes.append(route)
                    
                    #this checks if origin and destinations are in the same route   
                    if no_transfer_routes := [route for route in origin_routes if route in destination_routes]:
                        no_transfer_routes_lengths = []
                        for route in no_transfer_routes:
                            if route.index(origin) < route.index(destination):
                                no_transfer_routes_lengths.append(self.find_route_length(route[route.index(origin):route.index(destination)+1]))
                            else:
                                no_transfer_routes_lengths.append(self.find_route_length(route[route.index(destination):route.index(origin)+1]))
                                
                        minimum_length = min(no_transfer_routes_lengths)
                        tt+= demand*minimum_length
                        travel_times.extend([minimum_length for _ in range(demand)])
                        
                        
                        if final_grading: 
                            d0 += demand
                        
                    else:   

                        linking_route_lengths = []
                        
                        for route_o in origin_routes:
                            for route_des in destination_routes:
                                
                                if route_o_to_route_des := self.do_these_lists_share_elements(route_o, route_des):
                                    linking_route_lengths.append([self.find_route_using_transfer(origin, destination, route_o, route_des, route_o_to_route_des['shared_items'])[1], 1])

                                
                                else: 

                                    for link in all_routes:
                                        if (link != route_o) and (link != route_des):
                                            if (route_o_to_link := self.do_these_lists_share_elements(route_o, link)) and \
                                            (link_to_route_des := self.do_these_lists_share_elements(link, route_des)):
                                
                                                length = 10000000000000000000
                                                link_route = []
                                                for item_o in route_o_to_link['shared_items']:
                                                    for item_d in link_to_route_des['shared_items']:
                                                        if link.index(item_o) < link.index(item_d):
                                                            if (l := self.find_route_length((lr := link[link.index(item_o):link.index(item_d)+1]))) < length:
                                                                link_route = lr
                                                                length = l 
                                                                start = 0
                                                                end = -1

                                                                
                                                        else: 
                                                            if (l := self.find_route_length((lr := link[link.index(item_d):link.index(item_o)+1]))) < length:
                                                                link_route = lr
                                                                length = l   
                                                                start = -1
                                                                end = 0      
                                                    
                                                link_and_des = self.find_route_using_transfer(link_route[start], destination, link_route, route_des, [link_route[end]])[0]
                                                completed_route = self.find_route_using_transfer(origin, destination, route_o, link_and_des, [link_route[start]])
                                                linking_route_lengths.append([completed_route[1], 2])                    
                                                
                        if linking_route_lengths:
                            minimum = min([x[0] for x in linking_route_lengths])
                            for element in linking_route_lengths:
                                if element[0] == minimum:
                                    ttr += demand*element[1]
                                    tt += demand*element[0]
                                    travel_times.extend([(element[0]+(5*element[1])) for _ in range(demand)])   
                                    
                                    if final_grading: 
                                        if element[1] == 1: 
                                            d1 += demand
                                        elif element[1] == 2:
                                            d2 += demand 
                                        
                                            
                                    break
                        else:       
                            tu += demand   
                            if final_grading: 
                                dun += demand

                else:       
                    tu += demand 
                    if final_grading: 
                        dun += demand

                
        
                                
                                
            nodes_to_assess.pop(0)
            
        att = statistics.mean(travel_times)
        
        if final_grading: 
            d0 = (d0 / total_demand) * 100
            d1 = (d1 / total_demand) * 100
            d2 = (d2 / total_demand) * 100
            dun = (dun / total_demand) * 100
            
            return (int((tt + (5*ttr) + (att+50)*tu)), d0, d1, d2, dun, att)
        else: 
            return int((tt + (5*ttr) + (att+50)*tu))
            
        
    
    def evapourate_pheromone(self) -> None:
        """reduce the values in the pheromone matrix
        by a factore of E
        """
        for row in self.pheromone_matrix:
            for node in self.all_nodes: 
                new_p = row[node] * self.E
                if new_p > 1:
                    row[node] = new_p
                else: 
                    row[node] = 1
                    
    
    def lay_pheromone(self, ant : int) -> None:
        """increase the pheromone on a specific route 

        Args:
            ant (int): the best performing ant 
        """
        pheromone_update = ((self.Q*1000)/self.ant_metadata[ant]['fitness'])*100
        
        for index, node in enumerate(self.ant_metadata[ant]['path_tracker'][:-1]): 

            self.pheromone_matrix[node][self.ant_metadata[ant]['path_tracker'][index+1]] += pheromone_update
            self.pheromone_matrix[self.ant_metadata[ant]['path_tracker'][index+1]][node] += pheromone_update



    def edit_busyness_matrix(self) -> None:
    
        nodes_to_assess = [x for x in range(len(self.busyness_matrix))]
        for index, row in enumerate(self.normalised_busyness_matrix):
            for node in nodes_to_assess:
                if (node in self.bus_stops_connected) and (index in self.bus_stops_connected):
                    self.normalised_busyness_matrix[index][node] = 1
                    self.normalised_busyness_matrix[node][index] = 1
            nodes_to_assess.pop(0)   
    
    
    def reset_bus_route(self, route:int) -> None:
        
        edges = []
        for key in self.individual_edges.keys():
            if key != route: 
                edges.extend(self.individual_edges[key])
                   
        self.edge_tracker = list(set(edges))
        
        self.route_tracker[route] = []
        self.individual_edges[route] = []
        
        nodes_to_assess = [x for x in range(len(self.busyness_matrix))]
        all_routes = [x for x in self.route_tracker.values() if x]
        total_connected_nodes = np.unique([x for sublist in all_routes for x in sublist])
        self.bus_stops_connected = list(total_connected_nodes)
        tempary_normalised_busyness_matrix = np.reshape(self.scale_features(np.array(self.busyness_matrix).flatten()), np.shape(self.busyness_matrix))
        
        for index_d, row in enumerate(self.busyness_matrix):
            destination = index_d
            for node_o in nodes_to_assess:
                origin = node_o
                if (destination not in total_connected_nodes) and (origin not in total_connected_nodes):
                    self.normalised_busyness_matrix[index_d][node_o] = tempary_normalised_busyness_matrix[index_d][node_o]
                    
                        
           
        nodes_to_assess.pop(0)        
        
        
    
    
    def run_algorithm(self) -> None:
        self.logger.info(f'\n\nnew run ------------------------------------------------------------------------------')
        self.logger.info(f'Data folder = {self.data_folder}')
        self.logger.info(f"K = {self.K}")
        self.logger.info(f"E = {self.E}")
        self.logger.info(f"Q = {self.Q}")
        self.logger.info(f"A = {self.A}")
        self.logger.info(f"B = {self.B}")
        self.logger.info(f"Y = {self.Y}")
        self.logger.info(f"X = {self.X}")
        self.logger.info(f"Number of Routes = {self.ROUTES}")
        self.logger.info(f"Max Route Length = {self.MAX_LENGTH}")
        self.logger.info(f"Stop clause = {self.STOP}")
        self.logger.info(f"Folds = {self.FOLDS}")
        self.logger.info(f"Recombination = {self.recombination}")
        
        start = time.time()
        
        hyperparameter_metrics = []
        recomb = 2 if self.recombination else 1 
        current_fitness = 0
        
        for fold in range(self.FOLDS):
            self.logger.info(f"\n\n------------ FOLD {fold} ----------------")
            for comb in range(recomb):
                for bus in range(self.ROUTES): 
                    recomb_logs = []
                    print(f'Computing bus route {bus}', end ='')
                    self.pheromone_matrix = np.ones(np.shape(self.distance_matrix))*self.mean_of_none_zero_distance_elements
                    
                    #if recombination
                    if comb > 0: 
                        placeholder_route = self.route_tracker[bus]
                        placeholder_edges = self.individual_edges[bus]
                        print(f"recombination activated - reseting bus route {bus}")
                        self.reset_bus_route(bus)   
                                   
                    metrics = {'best_route' : [], 'best_fitness' : 100000000000, 'ant': None}   
                    start_node = self.find_start_node()
                    iterations_since_improvement = 0
                    iteration = 0 

                    while iterations_since_improvement < self.STOP:
                        
                        
                        for ant in range(self.K):
                            print(".", end='')
                            #clear any old metadata, set up for new iteration
                            self.ant_metadata[ant] = {
                            'start_node' : start_node, 
                            'current_node' : start_node,  
                            'path_tracker' : [start_node],
                            'i_have_been_stuck_once_already' : False
                            } 
                            
                            
                            while True:
                                next_node = self.pick_next_node(ant)

                                if next_node < 0:
                                    break
                            
                                self.ant_metadata[ant]['current_node'] = next_node
                                self.ant_metadata[ant]['path_tracker'].append(next_node)
                                
                                if self.MAX_LENGTH:
                                    if len(self.ant_metadata[ant]['path_tracker']) >= self.MAX_LENGTH:
                                        break
                            
                            self.route_tracker[bus] = self.ant_metadata[ant]['path_tracker']
                            self.ant_metadata[ant]['fitness'] = self.grade_solution()
                            self.route_tracker[bus] = []
                            
                        self.evapourate_pheromone()
                        best_fitness = min((fitness := [self.ant_metadata[ant]['fitness'] for ant in self.ant_metadata]))
                        best_ant = fitness.index(best_fitness)
                        
                        iteration += 1
                        iterations_since_improvement += 1
                        
                        if best_fitness < metrics['best_fitness']:
                            metrics['best_fitness'] = best_fitness
                            metrics['ant'] = best_ant
                            metrics['best_route'] = self.ant_metadata[ant]['path_tracker']
                            iterations_since_improvement = 0
                        
                        
                        self.lay_pheromone(best_ant)
                     
                    #if recombination   
                    if comb > 0: 
                        #if failed
                        if best_fitness > current_fitness:
                            
                            self.individual_edges[bus] = placeholder_edges
                            #add edges back to main edges list 
                            self.route_tracker[bus] = placeholder_route
                            self.bus_stops_connected.extend(placeholder_route) 
                            self.bus_stops_connected = list(set(self.bus_stops_connected)) 
                             
                            for node in placeholder_edges:
                                if node not in self.edge_tracker:
                                    self.edge_tracker.append(node)
                                if (node[1],node[0]) not in self.edge_tracker:
                                    self.edge_tracker.append((node[1],node[0]))   
                        #if success
                        else:
                            recomb_logs.append(f'recombing for route {bus} succesfull with a {current_fitness - best_fitness} improvement')
                            if self.AVOID_EDGES_THAT_ARE_COVERED:
                                for index, node in enumerate(metrics['best_route'][:-1]):
                                    
                                    if (node, metrics['best_route'][index+1]) not in self.edge_tracker:
                                        self.edge_tracker.append((node, metrics['best_route'][index+1]))
                                        self.individual_edges[bus].append((node, metrics['best_route'][index+1]))
                                    
                                    if (metrics['best_route'][index+1], node) not in self.edge_tracker:  
                                        self.edge_tracker.append((metrics['best_route'][index+1], node))
                                        self.individual_edges[bus].append((node, metrics['best_route'][index+1]))
                                        
                                    self.route_tracker[bus] = metrics['best_route']
                                    self.bus_stops_connected.extend(metrics['best_route']) 
                                    self.bus_stops_connected = list(set(self.bus_stops_connected))  
                    else:
                        if self.AVOID_EDGES_THAT_ARE_COVERED:
                            for index, node in enumerate(metrics['best_route'][:-1]):
                                
                                if (node, metrics['best_route'][index+1]) not in self.edge_tracker:
                                    self.edge_tracker.append((node, metrics['best_route'][index+1]))
                                    
                                    if self.recombination:
                                        self.individual_edges[bus].append((node, metrics['best_route'][index+1]))
                                
                                if (metrics['best_route'][index+1], node) not in self.edge_tracker:  
                                    self.edge_tracker.append((metrics['best_route'][index+1], node))
                                    
                                    if self.recombination:
                                        self.individual_edges[bus].append((node, metrics['best_route'][index+1]))
                            
                        current_fitness = best_fitness
                        
                        self.route_tracker[bus] = metrics['best_route']
                        self.bus_stops_connected.extend(metrics['best_route']) 
                        self.bus_stops_connected = list(set(self.bus_stops_connected))   
                    
                    
                    self.edit_busyness_matrix()   
                    
                    for message in recomb_logs: 
                        self.logger.info(message)
                    
                    print(" complete")
                        
                final_grading = self.grade_solution(final_grading=True)
               
                

                for route in self.route_tracker.keys():
                    self.logger.info(f'bus route {route} : {self.route_tracker[route]}')
                    
                self.logger.info(f'with fitness {final_grading[0]}')
                self.logger.info(f'with d0 {final_grading[1]}')
                self.logger.info(f'with d1 {final_grading[2]}')
                self.logger.info(f'with d2 {final_grading[3]}')
                self.logger.info(f'with dun {final_grading[4]}')
                self.logger.info(f'with att {final_grading[5]}')
                
                end = time.time()
                self.logger.info(f"This fold took {end - start} seconds")
             
            if self.provide_graphs:   
                for key in self.route_tracker.keys():
                    route = self.route_tracker[key]
                    edges = []
                    for index, node in enumerate(route[:-1]):
                        edges.append((node, route[index+1]))
                    
                    self.draw_graph(edges, f"Bus route {key}", 'r', self.graph_dir+"fold_"+str(fold)+"/", self.route_tracker[key])
                    
                    
            
            
               
            hyperparameter_metrics.append(final_grading)


            
        #! change this, this isn't the most effcient way to do this, couple be just one list 
        return {
                "fitness" : statistics.mean([x[0] for x in hyperparameter_metrics]), 
                "d0" : statistics.mean([x[1] for x in hyperparameter_metrics]),
                "d1" : statistics.mean([x[2] for x in hyperparameter_metrics]),
                "d2" : statistics.mean([x[3] for x in hyperparameter_metrics]),
                "dun" : statistics.mean([x[4] for x in hyperparameter_metrics]),
                "att" : statistics.mean([x[5] for x in hyperparameter_metrics]),
                }
               
            
    def add_log(self, log):
        self.logger.info(log)

with open("./data/input.json", 'r') as f:
    inputs = json.load(f)
    
aco = ACO_TNDP(**inputs)
aco.load_graph_data()
aco.run_algorithm()
    

        