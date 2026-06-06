import random
import statistics
import math
import bisect

import datasetMaxwel as dt
import parameter as prameters
import random_guessing as rg


class AlgoritmaGenetikaMaxwell:
    """
    Algoritma Genetika murni untuk dataset Maxwell.

    Populasi awal dibangkitkan secara acak.
    Kode ini digunakan sebagai pembanding terhadap model GACO,
    terutama untuk mengukur kontribusi ACO pada kualitas inisialisasi populasi awal.

    Output utama:
    - Best Fitness Awal
    - Mean Fitness Awal
    - Standar Deviasi Fitness Awal
    - Generasi Menuju Fitness Terbaik
    - MAE Akhir
    """

    def __init__(self, parameterSetting):
        self.popsize = parameterSetting["popsize"]
        self.crossoverRate = parameterSetting["crossoverRate"]
        self.numOfDimension = parameterSetting["numOfDimension"]
        self.mutationRate = parameterSetting["mutationRate"]
        self.ranges = parameterSetting["ranges"]
        self.maxIter = parameterSetting["maxIter"]
        self.stoppingFitness = parameterSetting["stoppingFitness"]
        self.patience = parameterSetting.get("patience", 10)
        self.elite_k = parameterSetting.get("elite_k", 1)
        self.mutationSigma = parameterSetting.get("mutationSigma", 0.05)
        self.seed = parameterSetting.get("seed", None)

        # Mapping kolom dataset Maxwell
        self.vi_idx = parameterSetting.get("vi_idx", 0)
        self.actual_idx = parameterSetting.get("actual_idx", 2)
        self.effort_idx = parameterSetting.get("effort_idx", 4)

        if self.seed is not None:
            random.seed(self.seed)

    # ============================================================
    # Utility dasar GA
    # ============================================================

    def initialPopulasi(self):
        """
        Membentuk satu kromosom secara acak berdasarkan rentang parameter.
        """
        chromosome = []

        for lowBound, uppBound in self.ranges:
            chromosome.append(random.uniform(lowBound, uppBound))

        return chromosome

    def initialPopulation(self):
        """
        Membentuk populasi awal secara acak.
        """
        return [self.initialPopulasi() for _ in range(self.popsize)]

    def getDeceleration(self, chromosome):
        """
        Menghitung nilai deceleration.

        Untuk dataset Maxwell:
        - 7 parameter pertama adalah Friction Factors
        - 15 parameter berikutnya adalah Dynamic Factors

        Total dimensi = 22.
        """

        hasilFf = 1.0
        hasilDf = 1.0

        # 7 parameter pertama → FF
        for gene in chromosome[:7]:
            hasilFf *= max(gene, 1e-12)

        # 15 parameter berikutnya → DF
        for gene in chromosome[7:]:
            hasilDf *= max(gene, 1e-12)

        return hasilFf * hasilDf

    def calcAE(self, deceleration, vi, effort, actualEffort):
        """
        Menghitung Absolute Error dan estimated effort.

        Formula:
        V = VI ^ D
        estimated effort = effort / V
        AE = |actual effort - estimated effort|
        """

        if vi <= 0:
            return 1e18, 0.0

        try:
            v = vi ** deceleration
            estEffort = effort / v
            AE = abs(actualEffort - estEffort)
        except OverflowError:
            AE = 1e18
            estEffort = 0.0

        return AE, estEffort

    def evaluate(self, population, vi, effort, actualEffort):
        """
        Mengevaluasi populasi berdasarkan nilai AE.

        Output:
        List tuple (AE, chromosome, estimated effort)
        Diurutkan dari AE terkecil.
        """

        results = []

        for chromosome in population:
            deceleration = self.getDeceleration(chromosome)
            AE, estEffort = self.calcAE(deceleration, vi, effort, actualEffort)
            results.append((AE, chromosome, estEffort))

        results.sort(key=lambda x: x[0])

        return results

    def fitnessFromAEs(self, AEs):
        """
        Mengubah AE menjadi fitness.

        Fitness = 1 / (1 + AE)
        Semakin kecil AE, semakin besar fitness.
        """

        fitness_values = [1.0 / (1.0 + ae) for ae in AEs]
        total_fitness = sum(fitness_values)

        if total_fitness <= 0:
            return [1.0 / len(fitness_values)] * len(fitness_values)

        return [fit / total_fitness for fit in fitness_values]

    def populationFitnessStats(self, AEs):
        """
        Menghitung statistik fitness populasi awal.

        Indikator:
        - Best Fitness Awal
        - Mean Fitness Awal
        - Standar Deviasi Fitness Awal
        """

        fitness_values = [1.0 / (1.0 + ae) for ae in AEs]

        best_fitness = max(fitness_values)
        mean_fitness = statistics.mean(fitness_values)

        if len(fitness_values) > 1:
            std_fitness = statistics.stdev(fitness_values)
        else:
            std_fitness = 0.0

        return best_fitness, mean_fitness, std_fitness

    def roulette(self, weights, k):
        """
        Roulette wheel selection.
        """

        if not weights:
            return []

        weights = [max(0.0, weight) for weight in weights]
        total = sum(weights)

        if total == 0.0:
            n = len(weights)
            weights = [1.0 / n] * n
            total = 1.0

        cumulative = []
        running_sum = 0.0

        for weight in weights:
            running_sum += weight
            cumulative.append(running_sum / total)

        selected_indexes = []

        for _ in range(k):
            r = random.random()
            idx = bisect.bisect_left(cumulative, r)

            if idx < 0:
                idx = 0

            if idx >= len(weights):
                idx = len(weights) - 1

            selected_indexes.append(idx)

        return selected_indexes

    def makePairs(self, indexes):
        """
        Membentuk pasangan parent.
        """

        random.shuffle(indexes)

        return [
            (indexes[i], indexes[i + 1])
            for i in range(0, len(indexes) - 1, 2)
        ]

    def singlePointCrossover(self, parent1, parent2, cut):
        """
        Single point crossover.
        """

        cut = max(0, min(cut, self.numOfDimension - 2))

        child1 = parent1[:cut + 1] + parent2[cut + 1:]
        child2 = parent2[:cut + 1] + parent1[cut + 1:]

        return child1, child2

    def mutate(self, chromosome):
        """
        Gaussian mutation.

        Mutasi dilakukan pada setiap gen berdasarkan mutationRate.
        Nilai hasil mutasi tetap berada dalam rentang parameter.
        """

        mutated = chromosome[:]

        for i in range(self.numOfDimension):
            if random.random() < self.mutationRate:
                lb, ub = self.ranges[i]
                step = self.mutationSigma * (ub - lb)

                mutated[i] = mutated[i] + random.gauss(0.0, step)
                mutated[i] = max(lb, min(ub, mutated[i]))

        return mutated

    # ============================================================
    # Main process GA
    # ============================================================

    def mainAlgen(self):
        datas = dt.CetakDataset.maxwelDataset()

        aeBestChromosomes = []
        estEffortBestChromosomes = []
        actualEfforts = []

        # Logging untuk tabel kontribusi ACO/GA
        initial_best_fitness_results = []
        initial_mean_fitness_results = []
        initial_std_fitness_results = []
        best_generation_results = []

        for data in datas:
            vi = data[self.vi_idx]
            actualEffort = data[self.actual_idx]
            effort = data[self.effort_idx]

            if vi <= 0:
                continue

            # ====================================================
            # 1. Inisialisasi populasi awal secara acak
            # ====================================================
            population = self.initialPopulation()

            # ====================================================
            # 2. Evaluasi populasi awal
            # ====================================================
            scored = self.evaluate(population, vi, effort, actualEffort)

            initial_AEs = [ae for ae, _, _ in scored]

            best_fit_awal, mean_fit_awal, std_fit_awal = self.populationFitnessStats(initial_AEs)

            initial_best_fitness_results.append(best_fit_awal)
            initial_mean_fitness_results.append(mean_fit_awal)
            initial_std_fitness_results.append(std_fit_awal)

            best_AE, best_chromosome, best_estEffort = scored[0]

            best_generation = 0
            no_improve = 0

            # ====================================================
            # 3. Proses evolusi GA
            # ====================================================
            for gen in range(self.maxIter):
                # Elitism
                elites = [
                    scored[i][1][:]
                    for i in range(min(self.elite_k, len(scored)))
                ]

                AEs = [ae for ae, _, _ in scored]
                fitness = self.fitnessFromAEs(AEs)

                # Seleksi parent
                parent_indexes = self.roulette(
                    fitness,
                    (self.popsize // 2) * 2
                )

                pairs = self.makePairs(parent_indexes)

                # Crossover dan mutasi
                offspring = []

                for i, j in pairs:
                    parent1 = population[i][:]
                    parent2 = population[j][:]

                    if random.random() < self.crossoverRate:
                        cut = random.randint(1, self.numOfDimension - 2)
                        child1, child2 = self.singlePointCrossover(parent1, parent2, cut)
                    else:
                        child1, child2 = parent1[:], parent2[:]

                    offspring.append(self.mutate(child1))
                    offspring.append(self.mutate(child2))

                # Replacement
                pool = population + offspring
                scored_pool = self.evaluate(pool, vi, effort, actualEffort)

                new_population = elites[:]

                for _, chromosome, _ in scored_pool:
                    if len(new_population) >= self.popsize:
                        break

                    new_population.append(chromosome[:])

                population = new_population
                scored = self.evaluate(population, vi, effort, actualEffort)
  
                # Update solusi terbaik
                if scored[0][0] + 1e-12 < best_AE:
                    best_AE, best_chromosome, best_estEffort = scored[0]
                    best_generation = gen + 1
                    no_improve = 0
                else:
                    no_improve += 1

                # Early stopping
                if best_AE <= self.stoppingFitness or no_improve >= self.patience:
                    break

            best_generation_results.append(best_generation)
           
            # print(best_estEffort)
            aeBestChromosomes.append(best_AE)
            estEffortBestChromosomes.append(best_estEffort)
            actualEfforts.append(actualEffort)
            # print(actualEffort)

        MAE = (
            sum(aeBestChromosomes) / len(aeBestChromosomes)
            if aeBestChromosomes
            else float("inf")
        )

        return {
            "MAE": MAE,
            "AEs": aeBestChromosomes,
            "estEfforts": estEffortBestChromosomes,
            "actualEfforts": actualEfforts,

            "Best Fitness Awal": statistics.mean(initial_best_fitness_results),
            "Mean Fitness Awal": statistics.mean(initial_mean_fitness_results),
            "Standar Deviasi Fitness Awal": statistics.mean(initial_std_fitness_results),
            "Generasi Menuju Fitness Terbaik": statistics.mean(best_generation_results),

            "Best Fitness Awal Per Project": initial_best_fitness_results,
            "Mean Fitness Awal Per Project": initial_mean_fitness_results,
            "Standar Deviasi Fitness Awal Per Project": initial_std_fitness_results,
            "Generasi Terbaik Per Project": best_generation_results
        }


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    ranges = prameters.prameterFfDf.parameter

    parameterSetting = {
        # =========================
        # Parameter GA
        # =========================
        "popsize": 40,
        "crossoverRate": 0.25,
        "numOfDimension": len(ranges),
        "mutationRate": 0.05,
        "ranges": ranges,
        "maxIter": 60,
        "stoppingFitness": 0.03,
        "patience": 10,
        "elite_k": 1,
        "mutationSigma": 0.05,
        "seed": 42,

        # =========================
        # Mapping kolom dataset Maxwell
        # =========================
        "vi_idx": 0,
        "actual_idx": 2,
        "effort_idx": 4,
    }

    algen = AlgoritmaGenetikaMaxwell(parameterSetting)
    hasil = algen.mainAlgen()

    print("===== HASIL GA DATASET MAXWELL =====")
    print("Best Fitness Awal:", hasil["Best Fitness Awal"])
    print("Mean Fitness Awal:", hasil["Mean Fitness Awal"])
    print("Standar Deviasi Fitness Awal:", hasil["Standar Deviasi Fitness Awal"])
    print("Generasi Menuju Fitness Terbaik:", hasil["Generasi Menuju Fitness Terbaik"])
    print("MAE Akhir:", hasil["MAE"])

    # ========================================================
    # Evaluasi tambahan menggunakan random guessing
    # ========================================================
    MAE = hasil["MAE"]
    estEfforts = hasil["estEfforts"]
    actualEfforts = hasil["actualEfforts"]

    runs = 1000
    random_guessing_runner = rg.RandomGuessing(actualEfforts, runs)
    randomGuessing = random_guessing_runner.mainRandomGuessing()

    MAE_P0 = randomGuessing["MAE_P0"]
    SA = 1 - (MAE / MAE_P0)

    StDev_P0 = statistics.stdev(randomGuessing["estEffortP0s"])
    ES = (MAE_P0 - MAE) / StDev_P0 if StDev_P0 > 0 else float("inf")

    def mbre(actual, estimated):
        denominator = min(actual, estimated)
        if denominator == 0:
            denominator = 1e-12
        return abs(actual - estimated) / denominator

    def mibre(actual, estimated):
        denominator = max(actual, estimated)
        if denominator == 0:
            denominator = 1e-12
        return abs(actual - estimated) / denominator

    MBRE = sum(
        mbre(actual, estimated)
        for actual, estimated in zip(actualEfforts, estEfforts)
    ) / len(actualEfforts)

    MIBRE = sum(
        mibre(actual, estimated)
        for actual, estimated in zip(actualEfforts, estEfforts)
    ) / len(actualEfforts)

    # print("\n===== EVALUASI TAMBAHAN =====")
    # print("MAE GA:", MAE)
    # print("MAE P0 Random Guessing:", MAE_P0)
    # print("Standard Accuracy:", SA)
    # print("Effect Size:", ES)
    # print("MBRE:", MBRE)
    # print("MIBRE:", MIBRE)

    print("\n===== DATA UNTUK UJI WILCOXON =====")
    print("ae_algen_maxwell = [")
    for ae in hasil["AEs"]:
        print(f"    {ae},")
    print("]")