import random
import sys
import statistics
import parameter as pr
import datasetMaxwel as dt
import random_guessing as rg

class GACO:
    def __init__(self, parameterSetting):
        self.popsize = parameterSetting['popsize']
        self.crossoverRate = parameterSetting['crossoverRate']
        self.numOfDimension = parameterSetting['numOfDimension']
        self.mutationRate = parameterSetting['mutationRate']
        self.ranges = parameterSetting['ranges']
        self.maxIter = parameterSetting['maxIter']
        self.stoppingFitness = parameterSetting['stoppingFitness']

        # ACO-specific
        self.pheromones = [1.0 for _ in range(self.numOfDimension)]
        self.evaporation_rate = 0.1
        self.alpha = 1
        self.beta = 2

    def initial_population(self):
        population = []
        for _ in range(self.popsize):
            chromosome = [random.uniform(lb, ub) for lb, ub in self.ranges]
            population.append(chromosome)
        return population

    def get_deceleration(self, chromosome):
        hasilFf = 1
        for i in range(7):
            hasilFf *= chromosome[i]
        hasilDf = 1
        for i in range(7, 22):
            hasilDf *= chromosome[i]
        return hasilFf * hasilDf

    def calc_AE(self, deceleration, vi, effort, actual_effort):
        v = vi ** deceleration
        est_effort = effort / v
        AE = abs(actual_effort - est_effort)
        return AE, est_effort

    def evaluate_population(self, population, vi, effort, actual):
        evaluations = []
        for chrom in population:
            dec = self.get_deceleration(chrom)
            ae, est = self.calc_AE(dec, vi, effort, actual)
            evaluations.append((ae, chrom, est))
        evaluations.sort(key=lambda x: x[0])
        return evaluations

    def ant_selection(self, evaluations):
        prob_selection = []
        total = sum([(1 / (ae + 1e-6)) ** self.beta for ae, _, _ in evaluations])
        for ae, chrom, _ in evaluations:
            prob = ((1 / (ae + 1e-6)) ** self.beta) / total
            prob_selection.append(prob)

        selected = []
        for _ in range(self.popsize):
            r = random.random()
            cum_prob = 0.0
            for i, prob in enumerate(prob_selection):
                cum_prob += prob
                if r <= cum_prob:
                    selected.append(evaluations[i][1])
                    break
        return selected

    def crossover(self, p1, p2):
        if random.random() < self.crossoverRate:
            point = random.randint(1, self.numOfDimension - 2)
            return p1[:point] + p2[point:]
        return p1[:]

    def mutate(self, chromosome):
        for i in range(self.numOfDimension):
            if random.random() < self.mutationRate:
                lb, ub = self.ranges[i]
                chromosome[i] = random.uniform(lb, ub)
        return chromosome

    def update_pheromones(self, best):
        for i in range(self.numOfDimension):
            self.pheromones[i] = (1 - self.evaporation_rate) * self.pheromones[i] + best[i]

    def run(self):
        datas = dt.CetakDataset.maxwelDataset()
        ae_results = []
        est_results = []

        for data in datas:
            vi, _, actual, _, effort = data
            population = self.initial_population()

            for _ in range(self.maxIter):
                evaluations = self.evaluate_population(population, vi, effort, actual)
                best_ae, best_chrom, best_est = evaluations[0]
                if best_ae <= self.stoppingFitness:
                    break

                selected = self.ant_selection(evaluations)

                next_generation = []
                for i in range(0, len(selected), 2):
                    p1 = selected[i]
                    p2 = selected[i+1 if i+1 < len(selected) else 0]
                    child = self.crossover(p1, p2)
                    mutated = self.mutate(child)
                    next_generation.append(mutated)

                population = next_generation
                self.update_pheromones(best_chrom)

            evaluations = self.evaluate_population(population, vi, effort, actual)
            best_ae, _, best_est = evaluations[0]
            ae_results.append(best_ae)
            est_results.append(best_est)
            est_best = min(est_results)
            print(f"{est_best:.12f}".replace('.', ','))
            sys.exit()
            # print(f" Best AE: {best_ae}, Best Est Effort: {best_est}")
            # print(f"{best_est:.12f}".replace('.', ','))
        MAE = sum(ae_results) / len(ae_results)
        # print(MAE)
        return {'MAE': MAE, 'AEs': ae_results, 'estEfforts': est_results}


if __name__ == '__main__':
    ranges = pr.prameterFfDf.parameter
    parameterSetting = {
        "popsize": 40,
        "crossoverRate": 0.7,
        "numOfDimension": len(ranges),
        "mutationRate": 1 / len(ranges),
        "ranges": ranges,
        "maxIter": 60,
        "stoppingFitness": 0.03
    }

gaco = GACO(parameterSetting)
result = gaco.run()
MAE = result['MAE']
estEfforts = result['estEfforts']
# print("Final MAE:", result['MAE'])
# print("AEs:", AE)
# print('result estimasi : ', hasil)

# Evaluasi
# ========
runs = 1000
run = rg.RandomGuessing(estEfforts, runs)
randomGuessing = run.mainRandomGuessing()

MAE_P0 = randomGuessing['MAE_P0']
estEffortP0s = randomGuessing['estEffortP0s']
SA_P0 = 1 - (MAE / MAE_P0)

# a count standar deviasi. dari variabel ESTaePOs
# ===============================================
StDev_P0 = statistics.stdev(estEffortP0s)
ES = abs((MAE - MAE_P0) / StDev_P0)

# MBRE AND MIBRE
for estEffort in range(len(estEfforts)):
    minEstimated = min(estEfforts[estEffort],
                       randomGuessing['estEffortP0s'][estEffort])
    maxEstimated = max(estEfforts[estEffort],
                       randomGuessing['estEffortP0s'][estEffort])
    AE = abs(estEfforts[estEffort] - randomGuessing['estEffortP0s'][estEffort])
    aeMinEstimated = AE / minEstimated
    aeMaxEstimated = AE / maxEstimated
    # print('AE', AE, 'MAE_P0', MAE_P0, 'estEffortP0s', estEffortP0s, 'SA_P0', SA_P0, 'STDEV P0', StDev_P0,
    #       'ES', ES, 'AE MIN ESTIMATED', aeMinEstimated, 'AE MAX ESTIMATED', aeMaxEstimated)