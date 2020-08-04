import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os

class GambleGame(object):
    def __init__(self):
        super().__init__()
        # here if we init randomly, the values will not coverge like fig 4.3 in the book. why?
        # self.values = np.random.rand(101) 
        self.values = np.zeros(101)
        self.values[100] = 1.0
        self.values[0] = 0.0
        self.policy = np.zeros(101)
        self.value_save = None

        """*********************************
        here to adjust the parameters freely
        *********************************"""
        self.pH = 0.4
        self.theta = 1e-17      # the smallest precise float number
        self.mode = "smallest"  # the way to choose the best action when the values are the same
        self.save_figure = False    # switch for if save the figures

    def valueIter(self):
        delta = 0
        for money in range(1, 100):
            v = self.values[money]
            self.values[money] = self.getHighestValue(money)
            delta = max(delta, abs(v-self.values[money]))
        return delta
    
    def getHighestValue(self, state):
        """
        return the highest value can be achieved with the given state

        state: the money owned, 1~99
        """
        max_V = -9999
        for action in range(1, min(state, 100-state)+1):
            V = self.getValue(state, action)
            if V > max_V:
                max_V = V
        return max_V

    def getBestAction(self, state, mode='smallest'):
        """
        return the best action under the given state

        state: the money owned, 1~99
        mode: the way to pick the best action when the value is the same, "smallest", "biggest" or "random"
        """
        best_actions = []
        max_V = -9999.0
        for action in range(1, min(state, 100-state)+1):
            V = self.getValue(state, action)
            if abs(V - max_V) < 1e-15:  # here we treat 'equal' tolerantly due to the unideal precision, very important!
                best_actions.append(action)
            elif V > max_V:
                best_actions = [action]
                max_V = V
        
        best_action = None
        if mode == "smallest":
            best_action = min(best_actions)
        elif mode == "biggest":
            best_action = max(best_actions)
        elif mode == "random":
            best_action = random.choice(best_actions)
        
        return best_action

    def getValue(self, state, action):
        """
        return the value of the state-action tuple

        state: the money owned, 1~99
        action: stake, 1~min(state, 100-state)
        """
        value = 0
        value += self.pH * self.values[state + action]
        value += (1 - self.pH) * self.values[state - action]
        return value

    def train(self):
        count = 1
        print("start training...")
        while True:
            t1 = time.time()
            delta = self.valueIter()
            t2 = time.time()

            if count in [1, 2, 3, 32]:
                if count == 1:
                    self.value_save = np.array(self.values, copy=True).reshape((1,101))
                else:
                    self.value_save = np.r_[self.value_save, self.values.reshape((1,101))]

            print("iter {:d} ends with delta = {:.18f}\tusing {:.7f} s".format(count, delta, t2-t1))
            count += 1
            if delta < self.theta:
                self.value_save = np.r_[self.value_save, self.values.reshape((1,101))]
                break
        
        print("iteration stop, start calculating the policy")
        for state in range(1, 100):
            self.policy[state] = self.getBestAction(state, mode=self.mode)

        self.visualize()

    def visualize(self):
        # value estimation
        fig = plt.figure()
        x = [i for i in range(0, 101)]
        y1 = [self.value_save[0, i] for i in x]
        y2 = [self.value_save[1, i] for i in x]
        y3 = [self.value_save[2, i] for i in x]
        y32 = [self.value_save[3, i] for i in x]
        y_cov = [self.value_save[3, i] for i in x]

        plt.plot(x, y1, linewidth=0.5)
        plt.plot(x, y2, linewidth=0.5)
        plt.plot(x, y3, linewidth=0.5)
        plt.plot(x, y32, linewidth=0.5)
        plt.plot(x, y_cov, linewidth=0.5)
        plt.title("value estimation - " + self.mode)
        plt.legend(["iter 1", "iter 2", "iter 3", "iter32", "final"])

        if self.save_figure:
            fig.savefig("./figures/4-9/value_estimation_" + self.mode + ".png", bbox_inches='tight')
        plt.show()
        plt.close()

        # policy
        fig = plt.figure()
        x = [i for i in range(1, 100)]
        y = [self.policy[i] for i in x]
        plt.step(x, y)
        plt.title("policy - " + self.mode)

        if self.save_figure:
            fig.savefig("./figures/4-9/policy_" + self.mode + ".png", bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == '__main__':
    # make the directory
    if not os.path.exists("./figures/4-9/"):
        os.makedirs("./figures/4-9")
    if not os.path.exists("./model_save/4-9/"):
        os.makedirs("./model_save/4-9")
    
    game = GambleGame()
    game.train()