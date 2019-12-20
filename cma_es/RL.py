import os
import datetime
import gym
import numpy as np
import matplotlib.pyplot as plt
from es import CMAES
import pandas as pd
import string

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Agent:
    def __init__(self, x, y, layer1_nodes, layer2_nodes):
        self.input = np.zeros(x, dtype=np.float128)
        self.weights1 = np.zeros((x, layer1_nodes), dtype=np.float128)
        self.weights2 = np.zeros((layer1_nodes, layer2_nodes), dtype=np.float128)
        self.weights3 = np.zeros((layer2_nodes, y), dtype=np.float128)
        self.output = np.zeros(y, dtype=np.float128)

    def feedforward(self, x):
        self.input = x
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))

    def assignWeights(self, s):
        self.weights1 = s[0]
        self.weights2 = s[1]
        self.weights3 = s[2]


class RL:
    
    def __init__(self, D="DefaultDir", H1=64, H2=64, P=100, G=5000, S=50000, E="Pong-v0", wd=0.01, #weight decay initialized to 0.01
                 si=0.5):
        # HYPERPARAMETERS
        self.HL1 = H1
        self.HL2 = H2
        self.NPOP = P
        self.MAX_ITER = G
        self.STEPS = S

        self.dir = D

        # CONSTANTS
        
        self.env = gym.make(E)
        self.STATE_SIZE = self.env.observation_space.shape[0]
        self.ACTION_SIZE = self.env.action_space.n
        self.env.reset()

        # CMA
        NPARAMS = (self.STATE_SIZE * self.HL1) + (self.HL1 * self.HL2) + (self.HL2 * self.ACTION_SIZE)
        cma = CMAES(NPARAMS, popsize=self.NPOP, weight_decay=wd, sigma_init=si)
        self.FINAL = self.Engine(cma)

    # Function to initialize
    def decisions_env(self, name):
        if name == "TimePilot-ram-v0":
            return 10
        elif name == "Breakout-ram-v0":
            return 4
        return 10  # just rerturn TimePilot command as default
    
    def findHighest(self, NN_Output):
        NN_Temp = NN_Output
        NN_I = []
        xF = 0
        index = 0
        foundI = 0
        for xl in range(self.ACTION_SIZE + 1):
            for NN_O in NN_Temp:
                if xF < NN_O:
                    xF = NN_O
                    foundI = index
                index = index + 1
            NN_Temp[foundI] = -1
            NN_I.append(foundI)
            index = 0
            xF = 0
        return NN_I[0]

    def weightsCalc(self, s):
        x1 = np.asarray(s[:self.STATE_SIZE * self.HL1], dtype=np.float128)
        x2 = np.asarray(s[self.STATE_SIZE * self.HL1:self.STATE_SIZE * self.HL1 + self.HL1 * self.HL2], dtype=np.float128)
        x3 = np.asarray(s[self.STATE_SIZE * self.HL1 + self.HL1 * self.HL2:], dtype=np.float128)
        x1 = np.reshape(x1, (self.STATE_SIZE, self.HL1))
        x2 = np.reshape(x2, (self.HL1, self.HL2))
        x3 = np.reshape(x3, (self.HL2, self.ACTION_SIZE))
        return (x1, x2, x3)

    # runs the sim and tallies reward
    def Fitness(self, solution):
        a = Agent(self.STATE_SIZE, self.ACTION_SIZE, self.HL1, self.HL2)
        a.assignWeights(self.weightsCalc(solution))
        fitness = 0

        self.env.reset()
        first = True
        for i in range(self.STEPS):
            if first:
                obs = a.input
                first = False
            a.feedforward(obs)
            choice = list(a.output)
            action = self.findHighest(choice)
            obs, reward, done, info = self.env.step(action)
            # self.env.render()
            fitness = fitness + reward
            if done:
                self.env.close()
                break
        self.env.close()
        return fitness
    # This function communicates with the es-tools CMA object "solver"
    def Engine(self, solver):
        history = []
        dLogs = []
        word = None
        Word = "Start: {0}\n".format(str(self.MAX_ITER))
        print(Word)
        dLogs.append(Word)
        for j in range(self.MAX_ITER):
            solutions = solver.ask()  # Generates parameters based on distribution mean and covariance matrix
            fitness_list = np.zeros(solver.popsize)
            for k in range(solver.popsize):
                fitness_list[k] = self.Fitness(solutions[k])
            solver.tell(fitness_list)  # update distribution mean and covariance matrix
            result = solver.result()
            history.append(result[1])
            if (j + 1) % 100 == 0:
                Word = "Iteration {0}___{1}".format((j + 1), result[1])
                print(Word)
                dLogs.append(Word)
            print("{0} {1}".format(str(j), self.dir), flush=True)

        print("local optimum discovered: {0}\n".format(result[0]))
        Word = "fitness score: {0}".format(result[1])
        print(Word)
        dLogs.append(Word)
        self.env.close()
        return result[1]





TestRLA = []      
startP = 50     
startG = 500
envAtari = ["Pong-ram-v0"]
environments = len(envAtari)
testCases = 1
Cases = 1
currentLabelE = None
currentLabelTC = None
#number of environments we are testing on
for currentE in range(environments):

    currentLabelE = envAtari[currentE].split("-")[0]
    if not (os.path.exists(currentLabelE)):
        os.mkdir(currentLabelE)
    else:
        print("{0} already exists".format(currentLabelE))
    print("TEST SET on {0} environment".format(envAtari[currentE]))

    #(start, end). index at start = 1, index at the end = end - 1
    for tC in range(1, (testCases + 1)): #tC is used as the generation max multiplier per {test case set}
        currentLabelTC = string.ascii_uppercase[tC - 1]
        print("CASE {0} Set Time G: {1}\n".format(currentLabelTC, str(startG)))

        for i in range(1, (Cases + 1)):            #i is used as the population multiplier per {test case}
            caseRL = RL(D="{0}/_RL_{1}{2}".format(currentLabelE, currentLabelTC, str(i-1)), P=(startP * i), G=(startG * tC), E=envAtari[currentE])
            TestRLA.append((int(caseRL.FINAL), caseRL.HL1, caseRL.HL2, caseRL.NPOP, caseRL.MAX_ITER, caseRL.STEPS))

        TestRLA = []
