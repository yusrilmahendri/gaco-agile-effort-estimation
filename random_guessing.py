import sys
import random
from itertools import groupby

class RandomGuessing():
    def __init__(self, estEfforts, runs):
        self.estEfforts = estEfforts
        self.runs = runs

    def getIndex(self, P0sFrequencies, maxHasil):
        for i in range(len(P0sFrequencies)):
            if maxHasil == P0sFrequencies[i]:
                return i

    def mainRandomGuessing(self):
        numOfData = len(self.estEfforts)
        P_0s = []
        estEffortP0s = []
        for i in range(numOfData):
            estEffort = self.estEfforts[i]
            for _ in range(self.runs):
                randomIndex = random.randint(0, numOfData-1)
                P0 = self.estEfforts[randomIndex]
                while estEffort == P0:
                    randomIndex = random.randint(0, numOfData-1)
                    P0 = self.estEfforts[randomIndex]
                P_0s.append(P0)
            P0sFrequencies = [len(list(group))
                              for key, group in groupby(sorted(P_0s))]
            maxHasil = max(P0sFrequencies)
            estEffortP0 = P_0s[self.getIndex(P0sFrequencies, maxHasil)]
            estEffortP0s.append(estEffortP0)
            P_0s = []
        return {'MAE_P0': sum(estEffortP0s)/len(self.estEfforts), 'estEffortP0s':  estEffortP0s}