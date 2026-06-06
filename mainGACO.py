# gaco_init_maxwell.py

import math
import random
import statistics
import bisect
import sys
import datasetMaxwel as dt
import parameter as pr
import random_guessing as rg


class HybridGA_InitACO_Maxwell:
    """
    Hybrid Genetic Algorithm dengan Ant Colony Optimization untuk dataset Maxwell.

    ACO digunakan hanya pada tahap inisialisasi populasi awal.
    Setelah populasi awal terbentuk, proses evolusi dilakukan menggunakan GA.

    Output utama:
    - Best Fitness Awal
    - Mean Fitness Awal
    - Standar Deviasi Fitness Awal
    - Generasi Menuju Fitness Terbaik
    - MAE Akhir
    """

    def __init__(self, parameterSetting):
        # =========================
        # Parameter Genetic Algorithm
        # =========================
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

        # =========================
        # Parameter ACO untuk inisialisasi
        # =========================
        self.rho = parameterSetting.get("rho", 0.1)
        self.alpha = parameterSetting.get("alpha", 1.0)
        self.beta = parameterSetting.get("beta", 2.0)
        self.q0 = parameterSetting.get("q0", 0.5)
        self.tau_init = parameterSetting.get("tau_init", 1.0)
        self.tau_min = parameterSetting.get("tau_min", 1e-6)
        self.tau_max = parameterSetting.get("tau_max", 100.0)

        # =========================
        # Mapping kolom dataset Maxwell
        # =========================
        self.vi_idx = parameterSetting.get("vi_idx", 0)
        self.actual_idx = parameterSetting.get("actual_idx", 2)
        self.effort_idx = parameterSetting.get("effort_idx", 4)

        if self.seed is not None:
            random.seed(self.seed)

    # ============================================================
    # Utility Genetic Algorithm
    # ============================================================

    def _initial_chromosome(self):
        """
        Membentuk satu kromosom acak berdasarkan rentang parameter.
        """
        return [random.uniform(lb, ub) for (lb, ub) in self.ranges]

    def _lnD(self, chromosome):
        """
        Menghitung ln(D).

        Kromosom berisi kombinasi nilai FF dan DF.
        Karena D berasal dari perkalian seluruh gen, maka digunakan log
        untuk menjaga stabilitas numerik.
        """
        total = 0.0

        for gene in chromosome:
            gene = max(gene, 1e-12)
            total += math.log(gene)

        return total

    def _calc_AE(self, lnD, vi, effort, actualEffort):
        """
        Menghitung Absolute Error dan estimated effort.

        D = exp(lnD)
        V = vi ^ D
        estimated effort = effort / V
        """

        if vi <= 0:
            return 1e18, 0.0

        lnD_clamped = max(min(lnD, 50.0), -50.0)
        D = math.exp(lnD_clamped)

        lnV = D * math.log(vi)
        estimatedEffort = effort * math.exp(-lnV)

        AE = abs(actualEffort - estimatedEffort)

        return AE, estimatedEffort

    def _evaluate(self, population, vi, effort, actual):
        """
        Mengevaluasi seluruh kromosom dalam populasi.

        Output berupa list:
        (AE, chromosome, estimatedEffort)

        Diurutkan berdasarkan AE terkecil.
        """
        scored = []

        for chromosome in population:
            lnD = self._lnD(chromosome)
            ae, estimated = self._calc_AE(lnD, vi, effort, actual)
            scored.append((ae, chromosome, estimated))

        scored.sort(key=lambda x: x[0])

        return scored

    def _fitness_from_AEs(self, AEs):
        """
        Mengubah AE menjadi probabilitas fitness.

        Fitness = 1 / (1 + AE)
        Semakin kecil AE, semakin besar fitness.
        """
        fitness_values = [1.0 / (1.0 + ae) for ae in AEs]
        total_fitness = sum(fitness_values)

        if total_fitness <= 0:
            return [1.0 / len(fitness_values)] * len(fitness_values)

        return [fit / total_fitness for fit in fitness_values]

    def _population_fitness_stats(self, AEs):
        """
        Menghitung statistik fitness populasi awal.

        Indikator:
        - Best fitness awal
        - Mean fitness awal
        - Standar deviasi fitness awal
        """
        fitness_values = [1.0 / (1.0 + ae) for ae in AEs]

        best_fitness = max(fitness_values)
        mean_fitness = statistics.mean(fitness_values)

        if len(fitness_values) > 1:
            std_fitness = statistics.stdev(fitness_values)
        else:
            std_fitness = 0.0

        return best_fitness, mean_fitness, std_fitness

    def _roulette(self, weights, k):
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

    def _make_pairs(self, indexes):
        """
        Membentuk pasangan parent untuk proses crossover.
        """
        random.shuffle(indexes)

        return [
            (indexes[i], indexes[i + 1])
            for i in range(0, len(indexes) - 1, 2)
        ]

    def _single_point_crossover(self, parent1, parent2, cut):
        """
        Single point crossover.
        """
        cut = max(0, min(cut, self.numOfDimension - 2))

        child1 = parent1[:cut + 1] + parent2[cut + 1:]
        child2 = parent2[:cut + 1] + parent1[cut + 1:]

        return child1, child2

    def _mutate(self, chromosome):
        """
        Gaussian mutation.

        Mutasi dilakukan pada setiap gen berdasarkan mutationRate.
        Nilai gen tetap dibatasi pada range parameter.
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
    # ACO untuk inisialisasi populasi awal
    # ============================================================

    def _aco_initialize_population(self, vi, effort, actual, n_ants=30, n_iters=10, bins=10):
        """
        Membentuk populasi awal menggunakan ACO.

        Setiap dimensi kromosom dibagi ke dalam beberapa bin.
        Semut memilih nilai berdasarkan kombinasi feromon dan heuristik.
        Kandidat terbaik akan memperkuat feromon pada bin yang dipilih.
        """

        edges = []
        centers = []

        for lb, ub in self.ranges:
            step = (ub - lb) / bins

            current_edges = [lb + i * step for i in range(bins + 1)]
            current_centers = [
                0.5 * (current_edges[i] + current_edges[i + 1])
                for i in range(bins)
            ]

            edges.append(current_edges)
            centers.append(current_centers)

        tau = [
            [self.tau_init] * bins
            for _ in range(self.numOfDimension)
        ]

        pool = []

        for _ in range(n_iters):
            candidates = []

            for _ant in range(n_ants):
                chromosome = []

                for d in range(self.numOfDimension):
                    heuristic = [1.0 / bins] * bins

                    scores = [
                        (max(tau[d][b], self.tau_min) ** self.alpha)
                        * (max(heuristic[b], 1e-12) ** self.beta)
                        for b in range(bins)
                    ]

                    if random.random() < self.q0:
                        # Eksploitasi: pilih salah satu bin terbaik secara acak
                        # agar tidak bias ke bin pertama ketika skor sama.
                        max_score = max(scores)
                        best_bins = [
                            i for i, score in enumerate(scores)
                            if abs(score - max_score) < 1e-12
                        ]
                        selected_bin = random.choice(best_bins)
                    else:
                        total_score = sum(scores) or 1.0
                        r = random.random()
                        acc = 0.0
                        selected_bin = 0

                        for i, score in enumerate(scores):
                            acc += score / total_score

                            if r <= acc:
                                selected_bin = i
                                break

                    chromosome.append(centers[d][selected_bin])

                lnD = self._lnD(chromosome)
                ae, estimated = self._calc_AE(lnD, vi, effort, actual)

                candidates.append((ae, chromosome, estimated))

            # Evaporasi feromon
            for d in range(self.numOfDimension):
                for b in range(bins):
                    tau[d][b] = max(
                        self.tau_min,
                        (1.0 - self.rho) * tau[d][b]
                    )

            # Update feromon berdasarkan kandidat terbaik
            candidates.sort(key=lambda x: x[0])
            best_ae, best_chromosome, _ = candidates[0]

            delta = 1.0 / (1.0 + best_ae)

            for d, value in enumerate(best_chromosome):
                current_edges = edges[d]

                bin_idx = bisect.bisect_right(current_edges, value) - 1
                bin_idx = min(len(current_edges) - 2, max(0, bin_idx))

                tau[d][bin_idx] = min(
                    self.tau_max,
                    tau[d][bin_idx] + delta
                )

            pool.extend(candidates)

        # Ambil kandidat terbaik dari pool ACO sebagai populasi awal GA
        pool.sort(key=lambda x: x[0])

        initial_population = [
            chromosome
            for _, chromosome, _ in pool[:self.popsize]
        ]

        # Jika belum cukup, tambahkan kromosom acak
        while len(initial_population) < self.popsize:
            initial_population.append(self._initial_chromosome())

        return initial_population

    # ============================================================
    # Main process
    # ============================================================

    def run(self):
        rows = dt.CetakDataset.maxwelDataset()

        ae_results = []
        estimated_results = []
        actual_results = []

        initial_best_fitness_results = []
        initial_mean_fitness_results = []
        initial_std_fitness_results = []
        best_generation_results = []

        for row in rows:
            vi = row[self.vi_idx]
            actual = row[self.actual_idx]
            effort = row[self.effort_idx]

            if vi <= 0:
                continue

            # ====================================================
            # 1. Inisialisasi populasi awal menggunakan ACO
            # ====================================================
            population = self._aco_initialize_population(
                vi,
                effort,
                actual,
                n_ants=50,
                n_iters=25,
                bins=15
            )

            # ====================================================
            # 2. Evaluasi populasi awal
            # ====================================================
            scored = self._evaluate(population, vi, effort, actual)

            initial_AEs = [ae for ae, _, _ in scored]

            best_fit_awal, mean_fit_awal, std_fit_awal = self._population_fitness_stats(initial_AEs)

            initial_best_fitness_results.append(best_fit_awal)
            initial_mean_fitness_results.append(mean_fit_awal)
            initial_std_fitness_results.append(std_fit_awal)

            best_AE, best_chromosome, best_estimated = scored[0]

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
                fitness = self._fitness_from_AEs(AEs)

                # Seleksi parent
                parent_indexes = self._roulette(
                    fitness,
                    (self.popsize // 2) * 2
                )

                pairs = self._make_pairs(parent_indexes)

                # Crossover dan mutasi
                offspring = []

                for i, j in pairs:
                    parent1 = population[i][:]
                    parent2 = population[j][:]

                    if random.random() < self.crossoverRate:
                        cut = random.randint(1, self.numOfDimension - 2)
                        child1, child2 = self._single_point_crossover(parent1, parent2, cut)
                    else:
                        child1, child2 = parent1[:], parent2[:]

                    offspring.append(self._mutate(child1))
                    offspring.append(self._mutate(child2))

                # Replacement
                pool = population + offspring
                scored_pool = self._evaluate(pool, vi, effort, actual)

                new_population = elites[:]

                for _, chromosome, _ in scored_pool:
                    if len(new_population) >= self.popsize:
                        break

                    new_population.append(chromosome[:])

                population = new_population
                scored = self._evaluate(population, vi, effort, actual)

                # Update solusi terbaik
                if scored[0][0] + 1e-12 < best_AE:
                    best_AE, best_chromosome, best_estimated = scored[0]
                    best_generation = gen + 1
                    no_improve = 0
                else:
                    no_improve += 1

                # Early stopping
                if best_AE <= self.stoppingFitness or no_improve >= self.patience:
                    break

            best_generation_results.append(best_generation)
   
            # print(best_estimated)

            # print(actual)
            ae_results.append(best_AE)
            estimated_results.append(best_estimated)
     
            actual_results.append(actual)

        MAE = sum(ae_results) / len(ae_results) if ae_results else float("inf")

        return {
            "Best Fitness Awal": statistics.mean(initial_best_fitness_results),
            "Mean Fitness Awal": statistics.mean(initial_mean_fitness_results),
            "Standar Deviasi Fitness Awal": statistics.mean(initial_std_fitness_results),
            "Generasi Menuju Fitness Terbaik": statistics.mean(best_generation_results),
            "MAE": MAE,
            "AEs": ae_results,
            "estEfforts": estimated_results,
            "actualEfforts": actual_results,
            "Best Fitness Awal Per Project": initial_best_fitness_results,
            "Mean Fitness Awal Per Project": initial_mean_fitness_results,
            "Standar Deviasi Fitness Awal Per Project": initial_std_fitness_results,
            "Generasi Terbaik Per Project": best_generation_results
        }


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    ranges = pr.prameterFfDf.parameter

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
        # Parameter ACO
        # =========================
        "rho": 0.1,
        "alpha": 1.5,
        "beta": 1.0,
        "q0": 0.5,
        "tau_init": 1.0,
        "tau_min": 1e-6,
        "tau_max": 50.0,

        # =========================
        # Mapping kolom dataset Maxwell
        # =========================
        "vi_idx": 0,
        "actual_idx": 2,
        "effort_idx": 4,
    }

    gaco = HybridGA_InitACO_Maxwell(parameterSetting)
    result = gaco.run()

    print("===== HASIL GACO DATASET MAXWELL =====")
    print("Best Fitness Awal:", result["Best Fitness Awal"])
    print("Mean Fitness Awal:", result["Mean Fitness Awal"])
    print("Standar Deviasi Fitness Awal:", result["Standar Deviasi Fitness Awal"])
    print("Generasi Menuju Fitness Terbaik:", result["Generasi Menuju Fitness Terbaik"])
    print("MAE Akhir:", result["MAE"])

    # ========================================================
    # Evaluasi tambahan menggunakan random guessing
    # ========================================================
    MAE = result["MAE"]
    estimatedEfforts = result["estEfforts"]
    actualEfforts = result["actualEfforts"]

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
        for actual, estimated in zip(actualEfforts, estimatedEfforts)
    ) / len(actualEfforts)

    MIBRE = sum(
        mibre(actual, estimated)
        for actual, estimated in zip(actualEfforts, estimatedEfforts)
    ) / len(actualEfforts)

    # print("\n===== EVALUASI TAMBAHAN =====")
    # print("MAE GACO:", MAE)
    # print("MAE P0 Random Guessing:", MAE_P0)
    # print("Standard Accuracy:", SA)
    # print("Effect Size:", ES)
    # print("MBRE:", MBRE)
    # print("MIBRE:", MIBRE)

    print("\n===== DATA UNTUK UJI WILCOXON =====")
    print("ae_gaco_maxwell = [")
    for ae in result["AEs"]:
        print(f"    {ae},")
    print("]")