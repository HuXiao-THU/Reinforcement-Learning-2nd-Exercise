# This model runs very slow, each iteration may need up to 30 minutes

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import os

class CarRental(object):
    def __init__(self):
        super().__init__()
        self.values = np.zeros((21,21), dtype=np.int)
        self.policy = np.zeros((21,21), dtype=np.int)
        self.stateIter = [(num1, num2) for num1 in range(0,21) for num2 in range(0,21)]
        self.modelSaveFile = './model_save/4-7/model.pkl'

        if os.path.exists(self.modelSaveFile):
            with open(self.modelSaveFile, 'rb') as f:
                self.values = pickle.load(f)
                self.policy = pickle.load(f)
        
        """*********************************
        here to adjust the parameters freely
        *********************************"""
        self.theta = 10
        self.gamma = 0.9
        self.max_rental = 10

    def policyEvaluation(self):
        count = 0
        while True:
            delta = 0
            start_time = time.time()
            for state in self.stateIter:
                v = self.values[state[0], state[1]]
                self.values[state[0], state[1]] = self.updateValue(state)
                delta = max(delta, abs(v-self.values[state[0], state[1]]))
            count += 1
            end_time = time.time()
            print("policy evaluation iter ", str(count), " ends with delta = ", delta, "\tusing ", end_time-start_time, "s")
            if delta < self.theta:
                break
    
    def policyImprovement(self):
        print("start policy improvment...")
        start_time = time.time()
        policy_stable = True
        for state in self.stateIter:
            old_action = self.policy[state[0], state[1]]
            self.policy[state[0], state[1]] = self.getBestAction(state)
            if self.getQValue(state, old_action) != self.getQValue(state, self.policy[state[0], state[1]]):
                policy_stable = False
        end_time = time.time()
        print("policy improvement end! using ", end_time-start_time, " s")
        return policy_stable

    def updateValue(self, state):
        """
        return the value of the state

        state: a list of numbers of cars in the two stations, like [10, 15]
        """
        return self.getQValue(state, self.policy[state[0], state[1]])

    def getBestAction(self, state):
        """
        return the best action under the given state
        
        state: a list of numbers of cars in the two stations, like [10, 15]
        """
        best_action = 0
        max_Q = -9999

        for action in self.getLegalActions(state):
            Q = self.getQValue(state, action)
            if Q > max_Q:
                best_action = action
                max_Q = Q
        
        return best_action

    def getQValue(self, state, action):
        """
        return the q value of the state-action tuple
        
        state: a list of numbers of cars in the two stations, like [10, 15]
        action: LEGAL number of cars moved from 1 to 2
        """
        value = 0
        # here we do a simplification to speed up the algorithm
        for rental_1 in range(0, self.max_rental+1):
            for rental_2 in range(0, self.max_rental+1):
                for return_1 in range(0, self.max_rental+1):
                    for return_2 in range(0, self.max_rental+1):
                        r = 0.0
                        p = 1.0
                        p *= (3**rental_1) * math.exp(-3) / math.factorial(rental_1)
                        p *= (4**rental_2) * math.exp(-4) / math.factorial(rental_2)
                        p *= (3**return_1) * math.exp(-3) / math.factorial(return_1)
                        p *= (2**return_2) * math.exp(-2) / math.factorial(return_2)

                        # move cars
                        num_station1 = state[0] - action
                        num_station2 = state[1] + action
                        num_station1 = min(num_station1, 20)    # no more than 20 cars
                        num_station2 = min(num_station2, 20)

                        if num_station1 > 10:
                            r -= 4
                        if num_station2 > 10:
                            r -= 4

                        r -= 2 * abs(action)
                        if action > 0:  # the free offer of the stuff
                            r += 2

                        # first lend out cars
                        rent_out_1 = min(rental_1, num_station1)
                        rent_out_2 = min(rental_2, num_station2)
                        num_station1 -= rent_out_1
                        num_station2 -= rent_out_2
                        
                        r += 10 * (rent_out_1 + rent_out_2)

                        # then return the cars
                        num_station1 += return_1
                        num_station2 += return_2
                        num_station1 = min(num_station1, 20)    # no more than 20 cars
                        num_station2 = min(num_station2, 20)

                        value += p * (r + self.gamma * self.values[num_station1, num_station2])
        
        return value

    def getLegalActions(self, state):
        """
        return the legal actions in a list
        """
        actions = [i for i in range(-5, 6)]
        for action in actions:
            if action > state[0] or action < -state[1]:
                actions.remove(action)
        return actions

    def train(self):
        """train the model"""
        count = 0
        self.visualize(count)
        while True:
            count += 1
            print("iter ", count, " start!")

            self.policyEvaluation()
            stable = self.policyImprovement()

            self.visualize(count)
            self.saveModel()

            if stable:
                print("policy has been stable!")
                break
            else:
                print("iter ", count, " end!")

    def visualize(self, count):
        """visualize the result"""
        fig = plt.figure()
        sns_plot = sns.heatmap(self.policy, cmap='bwr')
        sns_plot.invert_yaxis()
        plt.title("Policy " + str(count))
        # plt.show()
        fig.savefig("./figures/4-7/policy" + str(count) + ".png", bbox_inches='tight')
    
    def saveModel(self):
        """save the model in a pickle"""
        with open(self.modelSaveFile, 'wb') as f:
            pickle.dump(self.values, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.policy, f, pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    # make the directory
    if not os.path.exists("./figures/4-7/"):
        os.makedirs("./figures/4-7")
    if not os.path.exists("./model_save/4-7/"):
        os.makedirs("./model_save/4-7")

    problem = CarRental()
    problem.train()