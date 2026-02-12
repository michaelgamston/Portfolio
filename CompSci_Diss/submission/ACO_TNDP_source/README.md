## How to Run the algorithm 
Run the project using the terminal whilst in the ACO_TNDP_source directory. 
The Algorithm with look for the inputs in data/input.json. If you wish to customise any inputs, change them here. 
It will then look for geospatial and traveller data in data/graph_data.
Please read the blew sections for information on how to properly prepare data and inputs.

## Inputs
In /data look for a file called inputs.json. In here you will find a json document containing all the input parameters. 
Some default parameters have been placed in here, but please feel free to change them. 
This document looks like this: Comments are marked with a - 

{ 
    "custom_logging_filepath" : "output.log",     - must be a string or null, changes the log file name
    "custom_data_path" : "data/graph_data",       - must be a string or null, changes the location the algorithm looks in for the graph data 
    "k" : 4,                                      - must be 0 or higher 
    "e" : 0.8,                                    - must be 0 or higher 
    "q" : 10,                                     - must be 0 or higher
    "a" : 10,                                     - must be 0 or higher
    "b" : 1,                                      - must be 0 or higher
    "y" : 2,                                      - must be 0 or higher
    "x" : 1,                                      - must be 0 or higher
    "num_routes" : 8,                             - must be 1 or higher
    "max_length" : null,                          - must be 1 or higher or null
    "stop" : 3,                                   - must be 1 or higher
    "folds" : 1,                                  - must be 1 or higher
    "locations_provided" : true,                  - must be true or false 
    "recombination" : false,                      - must be true or false 
    "provide_graphs" : true                       - must be true or false 
}

## Graph Data

Defualt graph data is included. A copy of Mandl's network can be found at data/graph_data/ . 
If you have your own data you wish to test you are able to edit the files in data/graph_data/ but keep the file names the same! 

The structure of the data must be: 

Coords.txt: 
    - This contains the coordinates of the nodes in the network.
    - One entry per line. Do NOT space lines apart with blank lines. 
    - Each entry must only contain two numbers, x and y. Seperated by a space.
    - This is an optional file, if it is not included then the input parameters must reflect it.

Demand.txt and Visibility.txt:
    - These contained the traveller demand and the visibility between each node.
    - Each line represents one nodes relation to all other nodes.
    - One entry per line. Do NOT space lines apart with blank lines.
    - Each line must contain the same number of data points. 
    - The matrices must be syemtrical. 
    - Demand must only have intergers in it
    - Visibility must only have intergers or 'Inf'
    - 'Inf' is used in place when one node has no relation to another
    - All intergers must be 0 or higher 

## Outputs 

The algorithm will create an output folder. In the fold you will (if you have selected provide_graphs) find a graphed version of the complete problem space, 
and a folder for each fold the algorithm has run, in each of these folders you will each bus route the algorithm has created. 

A log file will also be produced in the same directory as the executable. This will be named based on what the input is for custom_logging_filepath.
If nothing is inputed it will be called 'ACO_TNDP_output.txt'.