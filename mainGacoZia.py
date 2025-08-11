import random
import datasetZia as dataset
import ffdfZia as prameter
import sys

class GACO:
    def __init__(self, setting):
        self.popsize = setting['popsize']
        self.crossoverRate = setting['crossoverRate']
        self.numOfDimension = setting['numOfDimension']
        self.mutationRate = setting['mutationRate']
        self.ranges = setting['ranges']
        self.maxIter = setting['maxIter']
        self.stoppingFitness = setting['stoppingFitness']
        self.pheromone = [1.0] * self.numOfDimension
        self.rho = 0.1  # pheromone evaporation rate

    def init_population(self):
        return [[random.uniform(lb, ub) for lb, ub in self.ranges] for _ in range(self.popsize)]

    def get_deceleration(self, chrom):
        hasilFf = chrom[0] * chrom[1] * chrom[2] * chrom[3]
        hasilDf = 1
        for i in range(4, 13):
            hasilDf *= chrom[i]
        return hasilFf * hasilDf

    def calc_AE(self, d, vi, effort, actual):
        v = vi**d
        est = effort / v
        return abs(actual - est), est

    def roulette_selection(self, AEs):
        inv = [1 / (ae + 1e-9) for ae in AEs]
        total = sum(inv)
        probs = [x / total for x in inv]
        cum_probs = [sum(probs[:i + 1]) for i in range(len(probs))]
        r = random.random()
        for i, cp in enumerate(cum_probs):
            if r <= cp:
                return i
        return len(probs) - 1

    def select_with_aco(self, population, AEs):
        probs = []
        for i, chrom in enumerate(population):
            tau = sum(self.pheromone) / len(self.pheromone)
            eta = 1 / (AEs[i] + 1e-6)
            probs.append(tau * eta)
        total = sum(probs)
        probs = [p / total for p in probs]
        cum_probs = [sum(probs[:i + 1]) for i in range(len(probs))]
        selected = []
        for _ in range(self.popsize):
            r = random.random()
            for i, cp in enumerate(cum_probs):
                if r <= cp:
                    selected.append(population[i])
                    break
        return selected

    def crossover(self, parents):
        offspring = []
        for _ in range(self.popsize // 2):
            if random.random() < self.crossoverRate:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                point = random.randint(1, self.numOfDimension - 1)
                child1 = p1[:point] + p2[point:]
                child2 = p2[:point] + p1[point:]
                offspring.extend([child1, child2])
            else:
                offspring.extend(random.sample(parents, 2))
        return offspring[:self.popsize]

    def mutate(self, pop):
        for i in range(len(pop)):
            for j in range(self.numOfDimension):
                if random.random() < self.mutationRate:
                    lb, ub = self.ranges[j]
                    pop[i][j] = random.uniform(lb, ub)
        return pop

    def update_pheromone(self, best):
        for i in range(self.numOfDimension):
            self.pheromone[i] = (1 - self.rho) * self.pheromone[i] + best[i]

    def run(self, datas):
        ae_results = []
        est_efforts = []
        for data in datas:
            population = self.init_population()
            bestChrom = None
            bestAE = float('inf')
            bestEst = 0
            for _ in range(self.maxIter):
                AEs = []
                Ests = []
                for chrom in population:
                    dec = self.get_deceleration(chrom)
                    ae, est = self.calc_AE(dec, data[1], data[0], data[8])
                    AEs.append(ae)
                    Ests.append(est)
                min_ae = min(AEs)
                idx = AEs.index(min_ae)
                if min_ae < bestAE:
                    bestAE = min_ae
                    bestChrom = population[idx]
                    bestEst = Ests[idx]
                if bestAE <= self.stoppingFitness:
                    break
                selected = self.select_with_aco(population, AEs)
                offspring = self.crossover(selected)
                mutated = self.mutate(offspring)
                population = mutated
                self.update_pheromone(bestChrom)
            print(f"{bestEst:.4f}".replace('.', ','))
            ae_results.append(bestAE)
            est_efforts.append(bestEst)
        mae = sum(ae_results) / len(ae_results)
        return {'MAE': mae, 'AEs': ae_results, 'estEfforts': est_efforts}


# Parameter setting
ranges = prameter.prameterFfDf.parameter
parameterSetting = {
    "popsize": 40,
    "crossoverRate": 0.25,
    "numOfDimension": 13,
    "mutationRate": 1 / 13,
    "ranges": ranges,
    "maxIter": 60,
    "stoppingFitness": 0.03
}

if __name__ == '__main__':
    datas = dataset.CetakDataset.ziauddinDataset()
    gaco = GACO(parameterSetting)
    result = gaco.run(datas)
    # print("\nFINAL RESULT:")
    # print("MAE:", result['MAE'])