For randomly spwaning plastic: 
    Beach():
        init():#add the following
        max spawn rate = 100 / number of CT's
        min spawn rate = 100 / number of CT's + (numer of CT's * 10)


        Step(): #add the following 

            if: switch on
            yes 
                randomly generate new spawn rate 
                find an empty cell 
                add plastic to scheduler
                add plastic to grid

            step scheduler 

For updating the cleaning jobs list 

    add self.cleaning_tank_jobs in beach init()

    in LWRobot()
        add in avoid_static_object()

        if isintance(object, HeaavyPlastic):
            update list 

        add new fucntion 
        update cleaing tank list(pos)
            self.model.cleaing_tank_list.append(pos)


For bidding on jobs 
in cleaning tank class
    def bid(self, target) -> int:

        #robots who are full wont be apart of the bidding 
        bid = 5 - hopper_fullness

        distance_to_target = dist(self, target)

        #it's very important the robot has enough battery 
        if battery > distance_to_target + dist(target, charger):
            bid += 5
        else 
            bid -= 5  
        
        if distance_to_target < 5:
            bid += 5
        elif < 10:
            bid += 4
        elif < 20:
            bid += 3
        elif < 40:
            bid += 2
        elif < 80:
            bid += 1


        return bid 

    def assign_new_job(self, target):
        new_target = target 
        update tracker 
        state == SEARCHING 

in model.py
    class Bid():
        object = CleaningTank 
        bid = 0

in beach
call from step at the end of the round 

def auction: 
    if jobs:
        bidders = [cleaning_tanks.state == MOVING]
        
        for job in jobs:  
            bids = []
            if len(bidders) > 1:        
                for tank in bidders:
                    bids.append(Bid(tank, tank.bid(job)))
            elif bidders:
                tank.assign_new_job(job)
                jobs.pop(job)
                break
            else: 
                break 


            winning_bid = max([bid.bid for bid in bids])
            winning_bid.tank.assign_new_job(job)
            jobs.pop(job)