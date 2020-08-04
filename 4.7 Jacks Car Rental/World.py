""" 
    This is the world model can be used for other training methods
"""

import numpy as np

class world(object):
    """
    the world model
    """
    def __init__(self):
        super().__init__()
        self.num_station1 = 10
        self.num_station2 = 10
    
    def observation(self):
        """
        return the state
        """
        return [self.num_station1, self.num_station2]
        
    def getLegalActions(self):
        """
        return the legal actions in a list
        """
        actions = [i for i in range(-5, 6)]
        for action in actions:
            if action > self.num_station1 or action < -self.num_station2:
                actions.remove(action)
        return actions
        
    def step(self, action):
        """
        do the action and return the reward

        action: LEGAL number of cars moved from 1 to 2, chosen from getLegalAction()
        """
        self.num_station1 -= action
        self.num_station2 += action
        self.num_station1 = min(self.num_station1, 20)
        self.num_station2 = min(self.num_station2, 20)

        reward = -2 * abs(action)

        rental_1 = np.random.poisson(lam=3)
        return_1 = np.random.poisson(lam=3)
        rental_2 = np.random.poisson(lam=4)
        return_2 = np.random.poisson(lam=2)

        # first lend out
        rental_num_1 = min(rental_1, self.num_station1)
        rental_num_2 = min(rental_2, self.num_station2)
        self.num_station1 -= rental_num_1
        self.num_station2 -= rental_num_2
        reward += 10 * rental_num_1 + 10 * rental_num_2

        # then return
        self.num_station1 += return_1
        self.num_station2 += return_2
        self.num_station1 = min(self.num_station1, 20)
        self.num_station2 = min(self.num_station2, 20)
        