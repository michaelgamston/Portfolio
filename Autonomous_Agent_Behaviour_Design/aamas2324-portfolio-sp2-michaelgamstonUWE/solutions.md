**How to Run My Solution** 

To run my solution simply run the run.py file that is linked. The solution should then pop up in a web brower, the same as it has been doing in our practicals. 
Then select the speed, the number of light robots and the number of cleaning tanks.
Push start in the top right corner to begin the simulation.

[Run.py](run.py)

**A breif Description**

The bin is coloured grey and only one robot can access it at a time.
The charging station is coloured a light teal.
The cleaing tanks are black squares and collect large rubbish.
The light robots are blue and collect small rubbish. 
Plastic is orange, with heavy plastic being the bigger of the two.
The plam trees are green. 

When a light robot is getting full it will reduce it's speed and turn yellow, once it is full it will turn red and move towards the bin to be emptied, onced emptied it's speed will increase again and it will continue to search it's area.  

**Activity One** 

When the added_functionality switch is on the following changes happen: 

When a cleaning tank is low on charge it will turn light teal and head to the charging station.

Heavy plastic now spawns at a stochastic rate of between 600 and 100 steps. 

When light robots bump into heavy plastic they will update a tracker with its location. At the start of each step an auction for the plastic in the tracker begins. All cleaning tanks are in state MOVING are called to submit a bid. The bid is based off their battery level, the distance from the target, and the hopper space. The cleaning tank with the highest bid wins. 

Diagrams 
[Light Robot](state_diagrams/Light_Tank_State_Diagram_Activity_One.pdf)

[Cleaning Tanks](state_diagrams/Cleaning_Tank_State_Diagram_Activity_One.pdf)

**Activity Two**

I've added Rubbish Collection Robots

Please read my brief report my on what I have done : [Click here for the report](Activity_two_description.md)

Rubbish Collection Robot are a small grey squares 

Compact plastics are medium size orange squares

Diagrams 
[Light Robot](state_diagrams/Light_Tank_State_Diagram_Activity_Two.pdf)

[Cleaning Tank](state_diagrams/Cleaning_Tank_State_Diagram_Activity_Two.pdf)

[Rubbish Collection Robot](state_diagrams/Rubbish_Collection_Robot_State_Diagram_Activity_Two.pdf)

**Activity Three**

Please follow the below link to my analysis note book

[Click here for analysis Notebook](system_analysis.ipynb)

**PLEASE NOTE TO CLICK MORE PAGES AT THE BOTTOM TO SEE THE WHOLE PDF**
[Click here for analysis PDF](act_three_pdf/system_analysis.pdf)


**Activity Four**

[Click here for my report into Proximal Policy Optimization](activity_four.md)